#!/usr/bin/env python3
# hold_detection.py
#
# Universal Hold Detector
# - Supports manual color selection (for batch pipeline)
# - Supports automatic red detection (default)
# - Handles all hold colors

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# PARAMETERS - ADJUST THESE FOR YOUR HOLDS
# ==============================================================================

# Default RED hold colors (if no manual color provided)
DEFAULT_LAB = np.array([145, 165, 135], dtype=np.float32)
DEFAULT_HSV = np.array([175, 180, 150], dtype=np.float32)

# Color matching tolerances
LAB_TOLERANCE = 25          # Tight for precise color matching
HSV_HUE_TOLERANCE = 8       # VERY strict hue (prevent red/orange confusion)
HSV_SAT_TOLERANCE = 80      # Base tolerance (increased 30% for chalk)
HSV_VAL_TOLERANCE = 80      # Base tolerance (increased 20% for chalk)

# Contour filters
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 12000
MIN_SOLIDITY = 0.4  # Filter out very irregular shapes

# Morphology
MORPH_KERNEL_SIZE = 9
DILATE_ITERATIONS = 3

# Frame sampling
NUM_FRAMES = 10

# Contour grouping
GROUP_DISTANCE = 80

# Debug output
DEBUG = True

# ==============================================================================


def normalize_lab(image: np.ndarray) -> np.ndarray:
    """Normalize LAB color space with CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.merge((l, a, b))


def sample_frames_from_video(video_path: str, num_frames: int = NUM_FRAMES) -> List[np.ndarray]:
    """Sample frames evenly from middle portion of video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample from middle 70% (skip start/end for better quality)
    start_frame = int(total_frames * 0.15)
    end_frame = int(total_frames * 0.85)
    
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame.copy())
    
    cap.release()
    print(f"[hold_detection] Sampled {len(frames)} frames from video")
    return frames


def create_color_mask(
    frame: np.ndarray,
    target_lab: np.ndarray,
    target_hsv: np.ndarray,
    is_red: bool = False
) -> np.ndarray:
    """
    Create mask for holds of specific color.
    
    Args:
        frame: Input frame
        target_lab: Target color in LAB space
        target_hsv: Target color in HSV space
        is_red: Whether target is red (needs special wraparound handling)
    """
    lab = normalize_lab(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # LAB-based mask
    diff_lab = np.linalg.norm(lab - target_lab, axis=2)
    mask_lab = (diff_lab < LAB_TOLERANCE).astype(np.uint8) * 255
    
    # For chalky holds: also detect FADED versions (whiter, less saturated)
    # Chalk makes ANY hold: lighter (higher L), less saturated (lower a/b)
    faded_lab = target_lab.copy()
    faded_lab[0] = min(255, faded_lab[0] + 30)  # Lighter
    
    diff_faded = np.linalg.norm(lab - faded_lab, axis=2)
    mask_faded = (diff_faded < LAB_TOLERANCE * 1.5).astype(np.uint8) * 255
    
    # Combine original and faded masks
    mask_lab = cv2.bitwise_or(mask_lab, mask_faded)
    
    # HSV-based mask
    hue = hsv[:, :, 0].astype(float)
    sat = hsv[:, :, 1].astype(float)
    val = hsv[:, :, 2].astype(float)
    
    if is_red:
        # Special handling for red (wraps around at 0/180)
        hue_diff = np.abs(hue - target_hsv[0])
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        
        # STRICT: Only allow hues in TRUE red range (exclude orange/pink)
        # Red is 0-10 or 170-180 in HSV (very strict to avoid orange at 10-25)
        if target_hsv[0] < 90:  # Wrapping from high side (175-180 wraps to 0-10)
            # Target is in the "low red" range (0-10)
            red_hue_mask = ((hue <= 10) | (hue >= 170)).astype(np.uint8) * 255
        else:
            # Target is in the "high red" range (170-180)
            red_hue_mask = ((hue <= 10) | (hue >= 170)).astype(np.uint8) * 255
    else:
        hue_diff = np.abs(hue - target_hsv[0])
        red_hue_mask = 255  # No restriction for non-red colors
        
        # For non-red colors, ensure they don't bleed into adjacent hues
        # Create a hard boundary around the target hue
        hue_min = max(0, target_hsv[0] - HSV_HUE_TOLERANCE)
        hue_max = min(180, target_hsv[0] + HSV_HUE_TOLERANCE)
        red_hue_mask = ((hue >= hue_min) & (hue <= hue_max)).astype(np.uint8) * 255
    
    sat_diff = np.abs(sat - target_hsv[1])
    val_diff = np.abs(val - target_hsv[2])
    
    # For red/chalky holds: be more lenient with saturation/value
    # BUT add saturation minimum to distinguish red from orange/pink
    if is_red:
        sat_tolerance = HSV_SAT_TOLERANCE * 1.3  # 30% more lenient - handles chalk
        val_tolerance = HSV_VAL_TOLERANCE * 1.2  # 20% more lenient - handles chalk
        
        # CRITICAL: Red must have reasonably high saturation (not orange/pink)
        # Pure red: high saturation (>100)
        # Orange: moderate saturation (80-150)
        # We want to catch red but not orange
        min_saturation = max(80, target_hsv[1] - 60)  # Require decent saturation
        sat_check = (sat >= min_saturation)
    else:
        sat_tolerance = HSV_SAT_TOLERANCE * 1.3  # 30% more lenient - handles chalk
        val_tolerance = HSV_VAL_TOLERANCE * 1.2  # 20% more lenient - handles chalk
        sat_check = True  # No minimum for non-red colors
    
    mask_hsv = (
        (hue_diff < HSV_HUE_TOLERANCE) &
        (sat_diff < sat_tolerance) &
        (val_diff < val_tolerance) &
        sat_check
    ).astype(np.uint8) * 255
    
    # Apply red hue restriction if applicable
    if is_red:
        mask_hsv = cv2.bitwise_and(mask_hsv, red_hue_mask)
    
    # For red, also try simple HSV range (covers both ends)
    if is_red:
        # DISABLED - too broad, picks up orange/purple
        # Only use the precise LAB+HSV matching above
        pass
        
        # Combine masks
        combined = cv2.bitwise_and(mask_lab, mask_hsv)
    else:
        # Combine LAB and HSV
        combined = cv2.bitwise_and(mask_lab, mask_hsv)
    
    # Morphological operations
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.dilate(combined, kernel, iterations=DILATE_ITERATIONS)
    
    return combined


def create_composite_mask(
    frames: List[np.ndarray],
    target_lab: np.ndarray,
    target_hsv: np.ndarray,
    is_red: bool = False,
    debug_dir: Optional[Path] = None
) -> np.ndarray:
    """Create composite mask from multiple frames."""
    print("[hold_detection] Creating composite mask from frames...")
    
    composite = np.zeros(frames[0].shape[:2], dtype=np.uint8)
    
    for i, frame in enumerate(frames):
        mask = create_color_mask(frame, target_lab, target_hsv, is_red)
        composite = cv2.bitwise_or(composite, mask)
        
        if DEBUG and debug_dir:
            cv2.imwrite(str(debug_dir / f"mask_{i:02d}.jpg"), mask)
    
    if DEBUG and debug_dir:
        cv2.imwrite(str(debug_dir / "composite_mask.jpg"), composite)
    
    return composite


def filter_and_group_contours(
    mask: np.ndarray,
    reference_frame: np.ndarray
) -> Tuple[List[str], Dict[str, List[int]], np.ndarray]:
    """Filter contours by size/shape and group nearby ones into holds."""
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[hold_detection] Found {len(contours)} raw contours")
    
    # Filter by size and solidity
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Size filter
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        
        # Solidity filter (removes very irregular shapes like text)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < MIN_SOLIDITY:
                continue
        
        valid_contours.append(cnt)
    
    print(f"[hold_detection] {len(valid_contours)} valid contours after filtering")
    
    # Group nearby contours
    groups = []
    used = set()
    
    for i, cnt1 in enumerate(valid_contours):
        if i in used:
            continue
        
        M1 = cv2.moments(cnt1)
        if M1["m00"] == 0:
            continue
        
        center1 = np.array([M1["m10"] / M1["m00"], M1["m01"] / M1["m00"]])
        group = [cnt1]
        used.add(i)
        
        # Find nearby contours
        for j, cnt2 in enumerate(valid_contours):
            if j <= i or j in used:
                continue
            
            M2 = cv2.moments(cnt2)
            if M2["m00"] == 0:
                continue
            
            center2 = np.array([M2["m10"] / M2["m00"], M2["m01"] / M2["m00"]])
            
            if np.linalg.norm(center1 - center2) < GROUP_DISTANCE:
                group.append(cnt2)
                used.add(j)
        
        groups.append(group)
    
    print(f"[hold_detection] Grouped into {len(groups)} holds")
    
    # Create hold positions and visualization
    hold_ids = []
    hold_positions = {}
    vis = reference_frame.copy()
    
    for idx, group in enumerate(groups):
        # Merge contours in group
        merged = np.vstack(group)
        hull = cv2.convexHull(merged)
        
        # Get centroid
        M = cv2.moments(hull)
        if M["m00"] == 0:
            continue
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        hold_id = f"hold_{idx}"
        hold_ids.append(hold_id)
        hold_positions[hold_id] = [cx, cy]
        
        # Draw on visualization
        cv2.drawContours(vis, [hull], -1, (0, 255, 0), 3)
        cv2.circle(vis, (cx, cy), 10, (0, 255, 255), -1)
        cv2.putText(vis, hold_id, (cx + 15, cy - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return hold_ids, hold_positions, vis


def detect_holds_v2(
    video_path: str,
    manual_color_lab: Optional[np.ndarray] = None,
    manual_color_hsv: Optional[np.ndarray] = None,
    output_dir: str = "output"
) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Detect climbing holds from video.
    
    Args:
        video_path: Path to video file
        manual_color_lab: Optional manual color in LAB space
        manual_color_hsv: Optional manual color in HSV space
        output_dir: Output directory for results
    
    Returns:
        hold_ids: List of hold IDs
        hold_positions: Dict mapping hold_id to [x, y] position
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    debug_dir = output_path / "debug_hold_detection"
    if DEBUG:
        debug_dir.mkdir(exist_ok=True)
    
    # Determine target color
    if manual_color_lab is not None and manual_color_hsv is not None:
        target_lab = manual_color_lab
        target_hsv = manual_color_hsv
        print("="*70)
        print("HOLD DETECTOR (Manual Color)")
        print("="*70)
        print(f"Target LAB: {target_lab.astype(int)}")
        print(f"Target HSV: {target_hsv.astype(int)}")
    else:
        target_lab = DEFAULT_LAB
        target_hsv = DEFAULT_HSV
        print("="*70)
        print("HOLD DETECTOR (Default: RED)")
        print("="*70)
        print(f"Target LAB: {target_lab.astype(int)}")
        print(f"Target HSV: {target_hsv.astype(int)}")
    
    # Check if target is red (needs special handling)
    is_red = (target_hsv[0] < 15 or target_hsv[0] > 170)  # Stricter red range
    if is_red:
        print("Detected RED target - using wraparound handling")
    
    print(f"Tolerances: LAB={LAB_TOLERANCE}, HSV=({HSV_HUE_TOLERANCE}, {HSV_SAT_TOLERANCE}, {HSV_VAL_TOLERANCE})")
    print("="*70)
    
    # Sample frames
    frames = sample_frames_from_video(video_path, NUM_FRAMES)
    reference_frame = frames[len(frames) // 2]
    
    # Create composite mask
    composite = create_composite_mask(
        frames,
        target_lab,
        target_hsv,
        is_red,
        debug_dir if DEBUG else None
    )
    
    # Detect and group holds
    hold_ids, hold_positions, vis = filter_and_group_contours(composite, reference_frame)
    
    # Save outputs
    cv2.imwrite(str(output_path / "hold_mask_composite.jpg"), composite)
    cv2.imwrite(str(output_path / "holds_debug.jpg"), vis)
    
    holds_json = output_path / "hold_positions_auto.json"
    with open(holds_json, "w") as f:
        json.dump(hold_positions, f, indent=2)
    
    # Save detected color info
    color_info = {
        "target_lab": target_lab.tolist(),
        "target_hsv": target_hsv.tolist(),
        "is_red": bool(is_red),
        "num_holds": len(hold_positions)
    }
    with open(output_path / "detected_color.json", "w") as f:
        json.dump(color_info, f, indent=2)
    
    print("="*70)
    print(f"✅ DETECTED {len(hold_positions)} HOLDS")
    print("="*70)
    print(f"Saved to: {holds_json}")
    print(f"Visualization: {output_path / 'holds_debug.jpg'}")
    if DEBUG:
        print(f"Debug output: {debug_dir}/")
    
    return hold_ids, hold_positions


def build_holds_json_from_video(
    video_path: str,
    output_json: str = "output/hold_positions_auto.json",
    debug_image_out: str = "output/holds_debug.jpg",
    **kwargs
) -> str:
    """Wrapper for backward compatibility with existing pipeline."""
    detect_holds_v2(video_path, output_dir="output")
    return output_json


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect climbing holds from video")
    parser.add_argument("--video", "-v", required=True, help="Path to video file")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    
    args = parser.parse_args()
    detect_holds_v2(args.video, output_dir=args.output)