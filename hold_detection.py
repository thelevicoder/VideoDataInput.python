# hold_detection.py

import cv2
import numpy as np
import json
import os
from typing import List, Dict, Tuple

import mediapipe as mp

mp_pose = mp.solutions.pose

LAB_TOLERANCE = 18
HSV_TOLERANCE = (12, 40, 40)
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 8000
GROUP_DISTANCE_THRESHOLD = 50  # For merging fragments
MORPH_KERNEL = np.ones((7, 7), np.uint8)


def normalize_lab(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.merge((l, a, b))


def extract_hold_mask(image: np.ndarray, ref_lab: np.ndarray, ref_hsv: np.ndarray) -> np.ndarray:
    """
    Given a frame and reference colors in LAB and HSV, produce a binary mask
    of pixels that match that color within the given tolerances.
    """
    lab_image = normalize_lab(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # LAB distance
    diff_lab = np.linalg.norm(lab_image - ref_lab, axis=2)
    mask_lab = (diff_lab < LAB_TOLERANCE).astype(np.uint8) * 255

    # HSV distance with circular hue
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


def load_video_frames_sequential(video_path: str, num_frames: int = 5) -> List[np.ndarray]:
    """
    Read the video once and grab approx num_frames evenly spaced frames
    without random seeking, to avoid frame read failures.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")

    fractions = np.linspace(0.0, 1.0, num_frames)
    target_indices = [int(f * (total_frames - 1)) for f in fractions]
    target_indices = sorted(set(target_indices))
    target_set = set(target_indices)

    frames: List[np.ndarray] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if idx in target_set:
            frames.append(frame.copy())

        idx += 1
        if idx > target_indices[-1]:
            break

    cap.release()
    print(f"[hold_detection] Loaded {len(frames)} frames for composite holds (targets {target_indices})")
    return frames


def infer_ref_colors_from_wrist_patches(
    video_path: str,
    max_search_frames: int = 600,
    stride: int = 1,
    patch_radius: int = 25,
    k_clusters: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan early frames until we see a pose. For each wrist, take a patch around
    the wrist and run KMeans on HSV inside that patch. Pick the cluster that:
      - has enough pixels
      - is reasonably saturated and bright
      - is not in the skin hue band (roughly 5 to 40)
    Return the LAB and HSV center of that cluster.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    chosen_lab = None
    chosen_hsv = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while frame_idx < max_search_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                frame_idx += 1
                continue

            print(f"[hold_detection] Using pose frame {frame_idx} for color sampling")

            landmarks = results.pose_landmarks.landmark

            wrists = [
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST,
            ]

            lab_img = normalize_lab(frame)
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            best_score = -1.0
            best_lab_center = None
            best_hsv_center = None

            for w_idx in wrists:
                wr = landmarks[w_idx]
                if wr.visibility < 0.5:
                    continue

                cx = int(wr.x * w)
                cy = int(wr.y * h)

                x0 = max(0, cx - patch_radius)
                x1 = min(w, cx + patch_radius)
                y0 = max(0, cy - patch_radius)
                y1 = min(h, cy + patch_radius)

                if x1 <= x0 or y1 <= y0:
                    continue

                patch_hsv = hsv_img[y0:y1, x0:x1].reshape(-1, 3)
                patch_lab = lab_img[y0:y1, x0:x1].reshape(-1, 3)

                if patch_hsv.shape[0] < k_clusters * 10:
                    continue

                patch_hsv_f = patch_hsv.astype(np.float32)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                ret, labels, centers = cv2.kmeans(
                    patch_hsv_f,
                    k_clusters,
                    None,
                    criteria,
                    5,
                    cv2.KMEANS_PP_CENTERS,
                )
                if ret is None or centers is None:
                    continue

                centers = centers.astype(np.float32)
                labels = labels.flatten()

                for ci in range(k_clusters):
                    mask = labels == ci
                    count = np.count_nonzero(mask)
                    if count < patch_hsv.shape[0] * 0.05:
                        # too few pixels, probably noise
                        continue

                    cluster_hsv = centers[ci]
                    h_val, s_val, v_val = cluster_hsv

                    # Normalised scores
                    s_norm = s_val / 255.0
                    v_norm = v_val / 255.0

                    # Skin-ish hues 5-40 get penalized
                    if 5 <= h_val <= 40:
                        skin_penalty = 0.6
                    else:
                        skin_penalty = 0.0

                    # Prefer fairly saturated and bright clusters
                    score = s_norm * 0.6 + v_norm * 0.3 - skin_penalty

                    if score > best_score:
                        best_score = score
                        # Compute LAB center from pixels in that cluster
                        cluster_lab_vals = patch_lab[mask]
                        cluster_lab_center = np.mean(cluster_lab_vals, axis=0)
                        best_lab_center = cluster_lab_center
                        best_hsv_center = cluster_hsv

            if best_lab_center is not None and best_hsv_center is not None:
                chosen_lab = best_lab_center
                chosen_hsv = best_hsv_center
                break

            frame_idx += 1

    cap.release()

    if chosen_lab is None or chosen_hsv is None:
        raise RuntimeError("Could not infer reference color from climber wrists")

    print(f"[hold_detection] Wrist inferred LAB: {chosen_lab}, HSV: {chosen_hsv}")
    return chosen_lab, chosen_hsv


def fallback_kmeans_color(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    KMeans fallback if pose based color inference fails.
    Slight bias toward saturated, bright clusters.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = normalize_lab(frame)

    h, w = hsv.shape[:2]
    scale = 0.25
    small_hsv = cv2.resize(hsv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small_lab = cv2.resize(lab, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    data_hsv = small_hsv.reshape(-1, 3).astype(np.float32)
    data_lab = small_lab.reshape(-1, 3).astype(np.float32)

    K = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, labels, centers_hsv = cv2.kmeans(
        data_hsv,
        K,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    if ret is None or centers_hsv is None:
        raise RuntimeError("KMeans color clustering failed")

    centers_hsv = centers_hsv.astype(np.float32)
    labels = labels.flatten()

    # Prefer saturated, bright clusters
    s = centers_hsv[:, 1]
    v = centers_hsv[:, 2]
    scores = (s / 255.0) * 0.7 + (v / 255.0) * 0.3
    idx = int(np.argmax(scores))

    ref_hsv = centers_hsv[idx]

    centers_lab = np.zeros_like(centers_hsv)
    for i in range(K):
        mask = labels == i
        if np.any(mask):
            centers_lab[i] = np.mean(data_lab[mask], axis=0)
        else:
            centers_lab[i] = np.mean(data_lab, axis=0)

    ref_lab = centers_lab[idx]

    print(f"[hold_detection] Fallback KMeans LAB: {ref_lab}, HSV: {ref_hsv}")
    return ref_lab, ref_hsv


def detect_holds_from_video(
    video_path: str,
    ref_lab: np.ndarray = None,
    ref_hsv: np.ndarray = None,
    save_path: str = "output/hold_positions.json",
    overlay_path: str = "output/hold_overlay.jpg",
    composite_mask_path: str = "output/hold_mask_composite.jpg",
) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """
    Multi frame composite hold detection.

    If ref_lab / ref_hsv not provided:
        - scan early frames for pose
        - sample wrist patches
        - pick "hold color" cluster inside that patch
        - fallback to KMeans on first frame if needed

    Then:
        - load ~5 frames across the video
        - build composite mask for that color
        - group nearby contours into holds
        - save centers and debug images
    """
    # Step 1: color selection
    if ref_lab is None or ref_hsv is None:
        try:
            ref_lab, ref_hsv = infer_ref_colors_from_wrist_patches(video_path)
        except Exception as e:
            print(f"[hold_detection] Pose based color inference failed: {e}")
            print("[hold_detection] Falling back to KMeans on first frame")
            cap = cv2.VideoCapture(video_path)
            ok, first_frame = cap.read()
            cap.release()
            if not ok or first_frame is None:
                raise RuntimeError("Could not read first frame for KMeans fallback")
            ref_lab, ref_hsv = fallback_kmeans_color(first_frame)

    # Step 2: multi frame composite mask
    frames = load_video_frames_sequential(video_path, num_frames=5)
    if not frames:
        raise RuntimeError("No frames loaded from video for hold detection")

    composite_mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
    for frame in frames:
        mask = extract_hold_mask(frame, ref_lab, ref_hsv)
        composite_mask = cv2.bitwise_or(composite_mask, mask)

    os.makedirs(os.path.dirname(composite_mask_path) or ".", exist_ok=True)
    cv2.imwrite(composite_mask_path, composite_mask)

    contours, _ = cv2.findContours(composite_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grouped_contours = []
    used = set()

    for i, c1 in enumerate(contours):
        if i in used:
            continue

        area1 = cv2.contourArea(c1)
        if area1 < MIN_CONTOUR_AREA or area1 > MAX_CONTOUR_AREA:
            continue

        group = [c1]
        M1 = cv2.moments(c1)
        if M1["m00"] == 0:
            continue
        c1_center = np.array([M1["m10"] / M1["m00"], M1["m01"] / M1["m00"]])

        for j, c2 in enumerate(contours):
            if j <= i or j in used:
                continue

            area2 = cv2.contourArea(c2)
            if area2 < MIN_CONTOUR_AREA or area2 > MAX_CONTOUR_AREA:
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
    hold_positions: Dict[str, Tuple[int, int]] = {}

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
            cv2.putText(output_vis, hold_id, (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(overlay_path, output_vis)
    with open(save_path, "w") as f:
        json.dump(hold_positions, f, indent=2)

    print(f"[hold_detection] Saved {len(hold_positions)} hold positions to {save_path}")
    print(f"[hold_detection] Saved overlay to {overlay_path}")
    return hold_ids, hold_positions


def build_holds_json_from_video(
    video_path: str,
    output_json: str = "output/hold_positions_auto.json",
    debug_image_out: str = "output/holds_debug.jpg",
    sample_frame_index: int = 0,  # unused, kept for compatibility
) -> str:
    """
    Wrapper expected by run_climb_pipeline.py.
    """
    _, _holds = detect_holds_from_video(
        video_path,
        ref_lab=None,
        ref_hsv=None,
        save_path=output_json,
        overlay_path=debug_image_out,
        composite_mask_path="output/hold_mask_composite.jpg",
    )
    return output_json


if __name__ == "__main__":
    video = "Vids/climbVid.mov"
    build_holds_json_from_video(video)
