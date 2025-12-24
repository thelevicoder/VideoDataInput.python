# hold_detection_v2.py
#
# ROBUST HOLD DETECTION - Complete rewrite
#
# Features:
# - Multiple detection strategies
# - Better color sampling
# - Extensive debugging output
# - Manual color override
# - Adaptive thresholds
# - Better climber filtering

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import mediapipe as mp

mp_pose = mp.solutions.pose

# ==============================================================================
# TUNABLE PARAMETERS
# ==============================================================================

# Color matching tolerances (higher = more lenient)
LAB_TOLERANCE = 25              # Was 18 - increased for better matching
HSV_HUE_TOLERANCE = 15          # Hue is circular (0-180)
HSV_SAT_TOLERANCE = 50          # Saturation tolerance
HSV_VAL_TOLERANCE = 50          # Value/brightness tolerance

# Contour size filters (pixels squared)
MIN_CONTOUR_AREA = 150          # Smaller minimum for small holds
MAX_CONTOUR_AREA = 10000        # Larger maximum for volumes

# Morphology
MORPH_KERNEL_SIZE = 7
DILATE_ITERATIONS = 2

# Frame sampling
NUM_FRAMES_TO_SAMPLE = 7        # More frames for better composite

# Debugging
DEBUG = True                     # Set False to disable debug output

# ==============================================================================


def create_debug_dir():
    """Create debug output directory."""
    debug_dir = Path("output/debug_hold_detection")
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def normalize_lab(image: np.ndarray) -> np.ndarray:
    """Normalize LAB image with CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.merge((l, a, b))


def sample_frames_from_video(video_path: str, num_frames: int = NUM_FRAMES_TO_SAMPLE) -> List[np.ndarray]:
    """
    Sample frames evenly from video, avoiding start/end where climber might block holds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")
    
    # Sample from middle 80% of video (skip first/last 10%)
    start_frame = int(total_frames * 0.1)
    end_frame = int(total_frames * 0.9)
    
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame.copy())
            if DEBUG:
                print(f"  Sampled frame {idx}/{total_frames}")
    
    cap.release()
    
    if not frames:
        raise RuntimeError("Could not sample any frames from video")
    
    print(f"[hold_detection] Sampled {len(frames)} frames from video")
    return frames


def infer_hold_color_from_wrists(
    video_path: str,
    max_frames: int = 300,
    stride: int = 5
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sample hold color from climber's wrists using KMeans clustering.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    print("[hold_detection] Attempting wrist-based color detection...")
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        
        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            if not results.pose_landmarks:
                frame_idx += 1
                continue
            
            # Found pose - sample wrist patches
            lab_img = normalize_lab(frame)
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            landmarks = results.pose_landmarks.landmark
            wrists = [
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST,
            ]
            
            # Collect color samples from both wrists
            all_lab_samples = []
            all_hsv_samples = []
            
            for wrist_lm in wrists:
                wrist = landmarks[wrist_lm]
                if wrist.visibility < 0.5:
                    continue
                
                cx = int(wrist.x * w)
                cy = int(wrist.y * h)
                
                # Extract 30x30 patch
                patch_size = 15
                x1 = max(0, cx - patch_size)
                x2 = min(w, cx + patch_size)
                y1 = max(0, cy - patch_size)
                y2 = min(h, cy + patch_size)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                patch_lab = lab_img[y1:y2, x1:x2].reshape(-1, 3)
                patch_hsv = hsv_img[y1:y2, x1:x2].reshape(-1, 3)
                
                all_lab_samples.append(patch_lab)
                all_hsv_samples.append(patch_hsv)
            
            if len(all_lab_samples) >= 1:
                # Combine samples from both wrists
                combined_lab = np.vstack(all_lab_samples)
                combined_hsv = np.vstack(all_hsv_samples)
                
                # KMeans to find dominant non-skin color
                from sklearn.cluster import KMeans
                
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans.fit(combined_hsv.astype(np.float32))
                
                centers = kmeans.cluster_centers_
                
                # Find non-skin cluster (hue not in 0-40 range)
                best_center_idx = None
                best_score = -1
                
                for i, center in enumerate(centers):
                    h, s, v = center
                    
                    # Avoid skin tones (hue 0-40)
                    if 0 <= h <= 40:
                        continue
                    
                    # Prefer saturated, bright colors
                    score = (s / 255.0) * 0.7 + (v / 255.0) * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_center_idx = i
                
                if best_center_idx is not None:
                    # Get corresponding LAB center
                    hsv_labels = kmeans.labels_
                    mask = hsv_labels == best_center_idx
                    
                    if np.any(mask):
                        lab_center = np.mean(combined_lab[mask], axis=0)
                        hsv_center = centers[best_center_idx]
                        
                        cap.release()
                        
                        print(f"[hold_detection] ✓ Detected hold color from wrists:")
                        print(f"  LAB: {lab_center.astype(int)}")
                        print(f"  HSV: {hsv_center.astype(int)}")
                        
                        return lab_center, hsv_center
            
            frame_idx += 1
    
    cap.release()
    print("[hold_detection] ✗ Could not detect color from wrists")
    return None, None


def fallback_kmeans_color(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback: Use KMeans on entire frame to find most saturated color cluster.
    """
    print("[hold_detection] Using KMeans fallback color detection...")
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = normalize_lab(frame)
    
    # Downsample for speed
    scale = 0.2
    h, w = frame.shape[:2]
    small_hsv = cv2.resize(hsv, (int(w*scale), int(h*scale)))
    small_lab = cv2.resize(lab, (int(w*scale), int(h*scale)))
    
    # KMeans on HSV
    from sklearn.cluster import KMeans
    
    data_hsv = small_hsv.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(data_hsv)
    
    centers_hsv = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Find most saturated, bright cluster
    best_idx = None
    best_score = -1
    
    for i, center in enumerate(centers_hsv):
        h, s, v = center
        
        # Skip very dark or very light
        if v < 50 or v > 250:
            continue
        
        # Skip skin tones
        if 0 <= h <= 40:
            continue
        
        score = (s / 255.0) * 0.8 + (v / 255.0) * 0.2
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    if best_idx is None:
        # Just pick most saturated
        saturations = centers_hsv[:, 1]
        best_idx = int(np.argmax(saturations))
    
    # Get LAB center for this cluster
    mask = labels == best_idx
    data_lab = small_lab.reshape(-1, 3).astype(np.float32)
    lab_center = np.mean(data_lab[mask], axis=0)
    hsv_center = centers_hsv[best_idx]
    
    print(f"[hold_detection] ✓ KMeans fallback color:")
    print(f"  LAB: {lab_center.astype(int)}")
    print(f"  HSV: {hsv_center.astype(int)}")
    
    return lab_center, hsv_center


def create_color_mask(
    frame: np.ndarray,
    ref_lab: np.ndarray,
    ref_hsv: np.ndarray
) -> np.ndarray:
    """
    Create binary mask of pixels matching reference color.
    """
    lab_img = normalize_lab(frame)
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # LAB distance mask
    diff_lab = np.linalg.norm(lab_img - ref_lab, axis=2)
    mask_lab = (diff_lab < LAB_TOLERANCE).astype(np.uint8) * 255
    
    # HSV mask with circular hue
    hue_diff = np.abs(hsv_img[:, :, 0].astype(float) - ref_hsv[0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    
    sat_diff = np.abs(hsv_img[:, :, 1].astype(float) - ref_hsv[1])
    val_diff = np.abs(hsv_img[:, :, 2].astype(float) - ref_hsv[2])
    
    mask_hsv = (
        (hue_diff < HSV_HUE_TOLERANCE) &
        (sat_diff < HSV_SAT_TOLERANCE) &
        (val_diff < HSV_VAL_TOLERANCE)
    ).astype(np.uint8) * 255
    
    # Combine masks
    combined = cv2.bitwise_and(mask_lab, mask_hsv)
    
    # Morphology to clean up
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.dilate(combined, kernel, iterations=DILATE_ITERATIONS)
    
    return combined


def create_composite_mask(
    frames: List[np.ndarray],
    ref_lab: np.ndarray,
    ref_hsv: np.ndarray,
    debug_dir: Optional[Path] = None
) -> np.ndarray:
    """
    Create composite mask from multiple frames.
    """
    print(f"[hold_detection] Creating composite mask from {len(frames)} frames...")
    
    composite = np.zeros(frames[0].shape[:2], dtype=np.uint8)
    
    for i, frame in enumerate(frames):
        mask = create_color_mask(frame, ref_lab, ref_hsv)
        composite = cv2.bitwise_or(composite, mask)
        
        if DEBUG and debug_dir:
            cv2.imwrite(str(debug_dir / f"mask_frame_{i:02d}.jpg"), mask)
    
    if DEBUG and debug_dir:
        cv2.imwrite(str(debug_dir / "composite_mask.jpg"), composite)
    
    return composite


def filter_climber_from_mask(
    mask: np.ndarray,
    video_path: str,
    debug_dir: Optional[Path] = None
) -> np.ndarray:
    """
    Remove climber body parts from mask using pose detection.
    """
    print("[hold_detection] Filtering climber from mask...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return mask
    
    # Get middle frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return mask
    
    h, w = frame.shape[:2]
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5,
    ) as pose:
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if not results.segmentation_mask:
            return mask
        
        # Create person mask
        person_mask = results.segmentation_mask
        person_mask = (person_mask > 0.5).astype(np.uint8) * 255
        
        # Dilate person mask to be conservative
        kernel = np.ones((20, 20), np.uint8)
        person_mask = cv2.dilate(person_mask, kernel, iterations=2)
        
        # Remove person from hold mask
        filtered_mask = cv2.bitwise_and(mask, cv2.bitwise_not(person_mask))
        
        if DEBUG and debug_dir:
            cv2.imwrite(str(debug_dir / "person_mask.jpg"), person_mask)
            cv2.imwrite(str(debug_dir / "filtered_mask.jpg"), filtered_mask)
        
        print("[hold_detection] ✓ Filtered climber from mask")
        return filtered_mask
    
    return mask


def detect_holds_from_mask(
    mask: np.ndarray,
    ref_frame: np.ndarray,
    debug_dir: Optional[Path] = None
) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """
    Detect individual holds from the composite mask.
    """
    print("[hold_detection] Detecting holds from mask...")
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"[hold_detection] Found {len(contours)} initial contours")
    
    # Filter by size
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA:
            valid_contours.append(contour)
    
    print(f"[hold_detection] {len(valid_contours)} contours passed size filter")
    
    # Group nearby contours
    GROUP_DISTANCE = 60
    groups = []
    used = set()
    
    for i, c1 in enumerate(valid_contours):
        if i in used:
            continue
        
        M1 = cv2.moments(c1)
        if M1["m00"] == 0:
            continue
        
        c1_center = np.array([M1["m10"] / M1["m00"], M1["m01"] / M1["m00"]])
        group = [c1]
        used.add(i)
        
        for j, c2 in enumerate(valid_contours):
            if j <= i or j in used:
                continue
            
            M2 = cv2.moments(c2)
            if M2["m00"] == 0:
                continue
            
            c2_center = np.array([M2["m10"] / M2["m00"], M2["m01"] / M2["m00"]])
            
            if np.linalg.norm(c1_center - c2_center) < GROUP_DISTANCE:
                group.append(c2)
                used.add(j)
        
        groups.append(group)
    
    print(f"[hold_detection] Grouped into {len(groups)} holds")
    
    # Extract hold positions
    hold_ids = []
    hold_positions = {}
    
    vis = ref_frame.copy()
    
    for idx, group in enumerate(groups):
        # Merge contours in group
        merged = np.vstack(group)
        hull = cv2.convexHull(merged)
        
        M = cv2.moments(hull)
        if M["m00"] == 0:
            continue
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        hold_id = f"hold_{idx}"
        hold_ids.append(hold_id)
        hold_positions[hold_id] = [cx, cy]
        
        # Draw on visualization
        cv2.drawContours(vis, [hull], -1, (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 8, (0, 255, 255), -1)
        cv2.putText(vis, hold_id, (cx + 10, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if DEBUG and debug_dir:
        cv2.imwrite(str(debug_dir / "holds_visualization.jpg"), vis)
    
    return hold_ids, hold_positions, vis


def detect_holds_v2(
    video_path: str,
    manual_color_lab: Optional[np.ndarray] = None,
    manual_color_hsv: Optional[np.ndarray] = None,
    output_dir: str = "output"
) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """
    Main hold detection function - Version 2.
    
    Args:
        video_path: Path to climbing video
        manual_color_lab: Optional manual LAB color (overrides detection)
        manual_color_hsv: Optional manual HSV color (overrides detection)
        output_dir: Output directory for results
        
    Returns:
        (hold_ids, hold_positions)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    debug_dir = create_debug_dir() if DEBUG else None
    
    print("="*70)
    print("HOLD DETECTION V2")
    print("="*70)
    
    # Step 1: Sample frames
    frames = sample_frames_from_video(video_path)
    ref_frame = frames[len(frames) // 2]  # Middle frame for reference
    
    # Step 2: Determine hold color
    if manual_color_lab is not None and manual_color_hsv is not None:
        print("[hold_detection] Using manual color values")
        ref_lab = manual_color_lab
        ref_hsv = manual_color_hsv
    else:
        # Try wrist-based detection
        ref_lab, ref_hsv = infer_hold_color_from_wrists(video_path)
        
        # Fallback to KMeans
        if ref_lab is None or ref_hsv is None:
            ref_lab, ref_hsv = fallback_kmeans_color(ref_frame)
    
    # Save detected color for reference
    if debug_dir:
        color_info = {
            "LAB": ref_lab.tolist(),
            "HSV": ref_hsv.tolist()
        }
        with open(debug_dir / "detected_color.json", "w") as f:
            json.dump(color_info, f, indent=2)
    
    # Step 3: Create composite mask
    composite_mask = create_composite_mask(frames, ref_lab, ref_hsv, debug_dir)
    
    # Step 4: Filter climber
    filtered_mask = filter_climber_from_mask(composite_mask, video_path, debug_dir)
    
    # Step 5: Detect holds
    hold_ids, hold_positions, vis = detect_holds_from_mask(
        filtered_mask, ref_frame, debug_dir
    )
    
    # Save outputs
    cv2.imwrite(str(output_path / "hold_mask_composite.jpg"), composite_mask)
    cv2.imwrite(str(output_path / "hold_mask_filtered.jpg"), filtered_mask)
    cv2.imwrite(str(output_path / "holds_debug.jpg"), vis)
    
    holds_json = output_path / "hold_positions_auto.json"
    with open(holds_json, "w") as f:
        json.dump(hold_positions, f, indent=2)
    
    print("="*70)
    print(f"✅ DETECTED {len(hold_positions)} HOLDS")
    print("="*70)
    print(f"  Saved to: {holds_json}")
    print(f"  Visualization: {output_path / 'holds_debug.jpg'}")
    if debug_dir:
        print(f"  Debug files: {debug_dir}/")
    
    return hold_ids, hold_positions


def build_holds_json_from_video(
    video_path: str,
    output_json: str = "output/hold_positions_auto.json",
    debug_image_out: str = "output/holds_debug.jpg",
    **kwargs
) -> str:
    """
    Wrapper for compatibility with existing pipeline.
    """
    _, _ = detect_holds_v2(video_path, output_dir="output")
    return output_json


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust hold detection v2")
    parser.add_argument("--video", "-v", required=True, help="Video path")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    detect_holds_v2(args.video, output_dir=args.output)
