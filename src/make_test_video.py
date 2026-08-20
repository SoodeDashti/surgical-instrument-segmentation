# src/make_test_video.py
# Builds a short, browser-playable test video from consecutive frames in the test set,
# so the dashboard's video tab can be tested without a real video source.

import os
import cv2
import imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "images")
SPLIT_FILE = os.path.join(PROJECT_ROOT, "files_used_for_testing.txt")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "test_video.mp4")

with open(SPLIT_FILE) as f:
    test_ids = [line.strip() for line in f if line.strip()]

frame_paths = [os.path.join(IMG_DIR, f"{stem}.jpg") for stem in test_ids[:40]]
frame_paths = [p for p in frame_paths if os.path.exists(p)]

if not frame_paths:
    raise RuntimeError("No test frames found - check IMG_DIR and SPLIT_FILE paths.")

first_frame = cv2.imread(frame_paths[0])
h, w = first_frame.shape[:2]

writer = imageio.get_writer(OUT_PATH, fps=5, codec="libx264", quality=8)
for p in frame_paths:
    frame_bgr = cv2.imread(p)
    frame_bgr_resized = cv2.resize(frame_bgr, (w, h))
    frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)
    writer.append_data(frame_rgb)
writer.close()

print(f"Saved test video with {len(frame_paths)} frames to {OUT_PATH}")