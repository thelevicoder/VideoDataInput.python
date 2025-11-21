# hold_detection.py (updated with multi-frame composite hold detection)

import cv2
import numpy as np
import json
import os
from typing import List, Dict, Tuple

LAB_TOLERANCE = 20
HSV_TOLERANCE = (15, 50, 50)
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 8000
GROUP_DISTANCE_THRESHOLD = 50  # For merging fragments
MORPH_KERNEL = np.ones((7, 7), np.uint8)


def normalize_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.merge((l, a, b))


def extract_hold_mask(image: np.ndarray, ref_lab: np.ndarray, ref_hsv: np.ndarray) -> np.ndarray:
    lab_image = normalize_lab(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    diff_lab = np.linalg.norm(lab_image - ref_lab, axis=2)
    mask_lab = (diff_lab < LAB_TOLERANCE).astype(np.uint8) * 255

    hue_diff = np.abs(hsv_image[:, :, 0] - ref_hsv[0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    diff_s = np.abs(hsv_image[:, :, 1] - ref_hsv[1])
    diff_v = np.abs(hsv_image[:, :, 2] - ref_hsv[2])
    mask_hsv = ((hue_diff < HSV_TOLERANCE[0]) &
                (diff_s < HSV_TOLERANCE[1]) &
                (diff_v < HSV_TOLERANCE[2])).astype(np.uint8) * 255

    combined = cv2.bitwise_and(mask_lab, mask_hsv)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, MORPH_KERNEL)
    combined = cv2.dilate(combined, MORPH_KERNEL, iterations=2)
    return combined


def load_video_frames(video_path: str, num_frames: int = 5) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def detect_holds_from_video(video_path: str, ref_lab: np.ndarray, ref_hsv: np.ndarray, save_path: str = "output/hold_positions.json") -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    frames = load_video_frames(video_path, num_frames=5)
    print(f"📽️ Loaded {len(frames)} frames for hold detection...")

    composite_mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
    for frame in frames:
        mask = extract_hold_mask(frame, ref_lab, ref_hsv)
        composite_mask = cv2.bitwise_or(composite_mask, mask)

    # Optional debug visualization
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/hold_mask_composite.jpg", composite_mask)

    contours, _ = cv2.findContours(composite_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grouped_contours = []
    used = set()

    for i, c1 in enumerate(contours):
        if i in used:
            continue
        group = [c1]
        M1 = cv2.moments(c1)
        if M1["m00"] == 0:
            continue
        c1_center = np.array([M1["m10"] / M1["m00"], M1["m01"] / M1["m00"]])

        for j, c2 in enumerate(contours):
            if j <= i or j in used:
                continue
            M2 = cv2.moments(c2)
            if M2["m00"] == 0:
                continue
            c2_center = np.array([M2["m10"] / M2["m00"], M2["m01"] / M2["m00"]])
            if np.linalg.norm(c1_center - c2_center) < GROUP_DISTANCE_THRESHOLD:
                group.append(c2)
                used.add(j)
        used.add(i)
        grouped_contours.append(group)

    hold_ids = []
    hold_positions = {}
    output_vis = frames[0].copy()

    for idx, group in enumerate(grouped_contours):
        merged = np.vstack(group)
        hull = cv2.convexHull(merged)
        M = cv2.moments(hull)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            hold_id = f"contour_{idx}"
            hold_ids.append(hold_id)
            hold_positions[hold_id] = (cx, cy)
            cv2.drawContours(output_vis, [hull], -1, (0, 255, 0), 2)
            cv2.circle(output_vis, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(output_vis, hold_id, (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite("output/hold_overlay.jpg", output_vis)
    with open(save_path, "w") as f:
        json.dump(hold_positions, f, indent=2)
    print(f"✅ Saved {len(hold_positions)} hold positions to {save_path}")

    return hold_ids, hold_positions


if __name__ == "__main__":
    sample_lab = np.array([147, 108, 124])
    sample_hsv = np.array([88, 101, 149])
    detect_holds_from_video("Vids/climbVid.mov", sample_lab, sample_hsv)
