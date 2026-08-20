# src/dashboard.py
# Gradio dashboard for surgical instrument segmentation.
# Supports single-image prediction (with TTA + uncertainty) and video/frame-sequence
# prediction (overlay applied frame by frame, exported as a video).
# Runs entirely on the locally-saved model checkpoint - no training happens here.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import imageio
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")

IMG_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model once at startup ---
model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,  # loading our own trained weights, not ImageNet ones
    in_channels=3,
    classes=1,
    activation=None,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])


def tta_predict(img_tensor):
    """Run inference on original + 3 augmented versions, return mean and std."""
    preds = []
    with torch.no_grad():
        preds.append(torch.sigmoid(model(img_tensor)))

        flipped_h = torch.flip(img_tensor, dims=[3])
        pred_h = torch.sigmoid(model(flipped_h))
        preds.append(torch.flip(pred_h, dims=[3]))

        flipped_v = torch.flip(img_tensor, dims=[2])
        pred_v = torch.sigmoid(model(flipped_v))
        preds.append(torch.flip(pred_v, dims=[2]))

        rotated = torch.rot90(img_tensor, k=1, dims=[2, 3])
        pred_r = torch.sigmoid(model(rotated))
        preds.append(torch.rot90(pred_r, k=-1, dims=[2, 3]))

    stacked = torch.stack(preds, dim=0)
    mean_pred = stacked.mean(dim=0)
    uncertainty_map = stacked.std(dim=0)
    return mean_pred, uncertainty_map


def predict_frame(frame_rgb, use_tta=True):
    """Runs prediction on a single RGB frame (numpy array). Returns overlay and uncertainty heatmap."""
    h, w = frame_rgb.shape[:2]
    transformed = val_transform(image=frame_rgb, mask=np.zeros((h, w)))
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    if use_tta:
        mean_pred, uncertainty_map = tta_predict(input_tensor)
    else:
        with torch.no_grad():
            mean_pred = torch.sigmoid(model(input_tensor))
        uncertainty_map = torch.zeros_like(mean_pred)

    pred_mask = (mean_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    pred_mask_resized = cv2.resize(pred_mask, (w, h))

    overlay = frame_rgb.copy()
    overlay[pred_mask_resized > 0] = [255, 0, 0]
    blended = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)

    uncertainty_np = uncertainty_map.squeeze().cpu().numpy()
    uncertainty_resized = cv2.resize(uncertainty_np, (w, h))
    uncertainty_heatmap = cv2.applyColorMap(
        (uncertainty_resized / (uncertainty_resized.max() + 1e-8) * 255).astype(np.uint8),
        cv2.COLORMAP_HOT,
    )
    uncertainty_heatmap = cv2.cvtColor(uncertainty_heatmap, cv2.COLOR_BGR2RGB)

    return blended, uncertainty_heatmap


def process_image(image):
    """Gradio callback for the image tab."""
    if image is None:
        return None, None
    blended, uncertainty_heatmap = predict_frame(image, use_tta=True)
    return blended, uncertainty_heatmap


def process_video(video_path, max_frames=60):
    """Gradio callback for the video tab. Processes up to max_frames frames and writes an output video."""
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    frames_out = []
    count = 0

    while count < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        blended, _ = predict_frame(frame_rgb, use_tta=False)  # single-pass for speed on video
        frames_out.append(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        count += 1

    cap.release()

    if not frames_out:
        return None

    out_path = os.path.join(PROJECT_ROOT, "results", "dashboard_video_output.mp4")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    for f in frames_out:
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)  # frames_out is BGR; imageio expects RGB
        writer.append_data(f_rgb)
    writer.close()

    return out_path


with gr.Blocks(title="Surgical Instrument Segmentation") as demo:
    gr.Markdown(
        "# Surgical Instrument Segmentation\n"
        "U-Net (MobileNetV2 backbone) trained on Kvasir-Instrument. "
        "Upload a frame or a short video to see the predicted instrument mask, "
        "overlaid on the image, alongside a pixel-wise uncertainty map. "
        "This is a research/portfolio demo, not a clinical tool."
    )

    with gr.Tab("Single Image"):
        with gr.Row():
            img_input = gr.Image(label="Input frame", type="numpy")
            img_output = gr.Image(label="Predicted overlay (TTA-averaged)")
            uncertainty_output = gr.Image(label="Uncertainty map (brighter = less confident)")
        img_button = gr.Button("Predict")
        img_button.click(
            fn=process_image,
            inputs=img_input,
            outputs=[img_output, uncertainty_output],
        )

    with gr.Tab("Video"):
        gr.Markdown(
            "Processes up to the first 60 frames for speed. "
            "Uses single-pass prediction per frame (no TTA) to stay fast enough for a live-style preview."
        )
        video_input = gr.Video(label="Input video")
        video_output = gr.Video(label="Predicted overlay video")
        video_button = gr.Button("Process video")
        video_button.click(
            fn=process_video,
            inputs=video_input,
            outputs=video_output,
        )

if __name__ == "__main__":
    demo.launch(server_port=7861)