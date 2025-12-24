# auto_climb_pipeline.py

import cv2
import os

def extract_wall_frame(video_path: str) -> str:
    """Extracts the midpoint frame of the video (original behavior)."""
    print(f"Extracting wall frame from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame.")
    image_path = "wall_example.jpg"
    cv2.imwrite(image_path, frame)
    cap.release()
    print(f"Saved wall image as {image_path}")
    return image_path

def extract_video_frames(video_path: str, frame_percentages=[10, 40, 80]) -> list:
    """Extracts frames at specified % points from the video and returns saved image paths."""
    print(f"Extracting {len(frame_percentages)} frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int((pct / 100.0) * total_frames) for pct in frame_percentages]

    output_paths = []
    for i, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            fname = f"wall_frame_{i}.jpg"
            cv2.imwrite(fname, frame)
            output_paths.append(fname)
        else:
            print(f"Warning: Couldn't read frame {frame_num}")
    cap.release()
    return output_paths
