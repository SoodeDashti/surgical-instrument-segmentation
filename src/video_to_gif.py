# src/video_to_gif.py
# Converts a screen-recording of the live dashboard (upload + output visible
# together) into a GIF for embedding in the README.
# (GitHub Markdown auto-plays GIFs inline; it does not auto-play <video> tags.)

import os
import imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PATH = os.path.join(PROJECT_ROOT, "results", "dashboard_demo.mp4")
OUT_PATH = os.path.join(PROJECT_ROOT, "results", "dashboard_demo.gif")

# --- tuning knobs to keep file size reasonable ---
RESIZE_WIDTH = 500      # shrink width (keeps aspect ratio); set None to skip
FRAME_SKIP = 4          # keep 1 out of every N frames (4 = a quarter of the frames)

reader = imageio.get_reader(IN_PATH)
meta = reader.get_meta_data()
fps = meta["fps"] / FRAME_SKIP

writer = imageio.get_writer(OUT_PATH, fps=fps)

for i, frame in enumerate(reader):
    if i % FRAME_SKIP != 0:
        continue
    if RESIZE_WIDTH:
        from PIL import Image
        import numpy as np
        img = Image.fromarray(frame)
        w_percent = RESIZE_WIDTH / float(img.width)
        h_size = int(float(img.height) * w_percent)
        img = img.resize((RESIZE_WIDTH, h_size), Image.LANCZOS)
        frame = np.array(img)
    writer.append_data(frame)

writer.close()
print(f"Saved GIF to {OUT_PATH}")