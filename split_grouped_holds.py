#!/usr/bin/env python3
# split_grouped_holds.py
#
# Automatically detect and split holds that were incorrectly grouped together

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict

def analyze_hold_for_split(contour: np.ndarray, mask: np.ndarray, video_path: str = None) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Analyze a hold contour to determine if it should be split.
    Uses multiple video frames to avoid climber occlusion.
    CONSERVATIVE: Only splits obvious cases of multiple holds grouped together.
    
    Returns:
        should_split: Whether this hold should be split
        centers: List of center points if splitting
    """
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Don't split very small holds
    area = cv2.contourArea(contour)
    if area < 500:  # Reduced from 800 to catch smaller grouped holds
        return False, []
    
    # Removed aspect ratio filter - grouped holds can be any shape
    
    # Extract the region
    roi_mask = mask[y:y+h, x:x+w]
    
    # If we have video, sample multiple frames to get better mask
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample 5 different frames
            frame_indices = np.linspace(int(total_frames * 0.2), int(total_frames * 0.8), 5, dtype=int)
            
            combined_roi = np.zeros_like(roi_mask)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Extract same ROI from this frame
                    frame_roi = frame[y:y+h, x:x+w]
                    
                    # Simple thresholding to detect hold (not climber)
                    frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
                    _, frame_thresh = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Combine with mask
                    combined_roi = cv2.bitwise_or(combined_roi, cv2.bitwise_and(roi_mask, frame_thresh))
            
            cap.release()
            
            if combined_roi.sum() > 0:
                roi_mask = combined_roi
    
    # Use distance transform to find local maxima (hold centers)
    dist_transform = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    
    # Normalize
    if dist_transform.max() > 0:
        dist_norm = (dist_transform / dist_transform.max() * 255).astype(np.uint8)
    else:
        return False, []
    
    # Use 60% threshold to find more peaks (less conservative)
    threshold = 0.60 * dist_norm.max()  # Reduced from 0.75
    _, peaks = cv2.threshold(dist_norm, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphology to clean peaks - smaller kernel to preserve peaks
    kernel = np.ones((5, 5), np.uint8)  # Smaller kernel (was 7x7)
    peaks = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, kernel)
    
    # Find connected components in peaks
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks, connectivity=8)
    
    # Need at least 2 peaks (excluding background)
    num_peaks = num_labels - 1
    
    if num_peaks < 2:
        return False, []
    
    # Get peak centers (in original image coordinates)
    centers = []
    for i in range(1, num_labels):  # Skip background (0)
        # Only include peaks with reasonable size
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:  # Too small, noise
            continue
            
        cx, cy = centroids[i]
        # Convert to original image coordinates
        centers.append((int(x + cx), int(y + cy)))
    
    # Need at least 2 valid centers
    if len(centers) < 2:
        return False, []
    
    # Check separation - reduced to catch closer grouped holds
    min_separation = 35  # Reduced from 50
    
    max_dist = 0
    valid_pairs = 0
    
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            max_dist = max(max_dist, dist)
            if dist > min_separation:
                valid_pairs += 1
    
    # Split if we have reasonably separated centers
    if max_dist > min_separation and valid_pairs > 0:
        # Allow up to 4 holds per group
        if len(centers) > 4:  # Was 3
            return False, []
        return True, centers
    
    return False, []


def split_hold(
    contour: np.ndarray,
    mask: np.ndarray,
    centers: List[Tuple[int, int]],
    original_id: str
) -> List[Dict]:
    """
    Split a grouped hold into multiple holds using watershed algorithm.
    
    Returns:
        List of new hold dictionaries with ids and positions
    """
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract region
    roi_mask = mask[y:y+h, x:x+w]
    
    # Create marker image for watershed
    markers = np.zeros_like(roi_mask, dtype=np.int32)
    
    # Mark each center with a different label
    for i, (cx, cy) in enumerate(centers):
        # Convert to ROI coordinates
        roi_x = cx - x
        roi_y = cy - y
        
        # Mark a small region around each center
        cv2.circle(markers, (roi_x, roi_y), 5, i+1, -1)
    
    # Create 3-channel image for watershed
    roi_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    
    # Apply watershed
    cv2.watershed(roi_3ch, markers)
    
    # Extract each region
    new_holds = []
    
    for i in range(1, len(centers) + 1):
        # Get mask for this region
        region_mask = (markers == i).astype(np.uint8) * 255
        
        # Find contour for this region
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        # Get largest contour
        region_contour = max(contours, key=cv2.contourArea)
        
        # Convert back to original coordinates
        region_contour = region_contour + np.array([x, y])
        
        # Get centroid
        M = cv2.moments(region_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            new_holds.append({
                'id': f"{original_id}_split_{i}",
                'position': [cx, cy],
                'contour': region_contour
            })
    
    return new_holds


def split_grouped_holds(
    holds_json_path: str = "output/hold_positions_auto.json",
    mask_path: str = "output/hold_mask_composite.jpg",
    debug_image_path: str = "output/holds_debug.jpg",
    output_dir: str = "output",
    video_path: str = None
):
    """
    Analyze holds and split any that were incorrectly grouped together.
    """
    print("\n" + "="*70)
    print("HOLD SPLITTER - Detecting Grouped Holds")
    print("="*70)
    
    # Load holds
    with open(holds_json_path, 'r') as f:
        holds = json.load(f)
    
    print(f"Loaded {len(holds)} holds")
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"❌ Error: Could not load mask from {mask_path}")
        return
    
    # Get clean frame from video if provided, otherwise use debug image
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Get middle frame
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, clean_frame = cap.read()
            cap.release()
            
            if not ret:
                # Fallback to debug image
                clean_frame = cv2.imread(debug_image_path)
        else:
            clean_frame = cv2.imread(debug_image_path)
    else:
        clean_frame = cv2.imread(debug_image_path)
    
    if clean_frame is None:
        print(f"❌ Error: Could not load base image")
        return
    
    # Find contours in mask - filter by size to get actual holds
    all_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to only include holds (not noise)
    MIN_HOLD_AREA = 200  # Match hold_detection.py
    MAX_HOLD_AREA = 12000
    
    contours = []
    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        if MIN_HOLD_AREA <= area <= MAX_HOLD_AREA:
            contours.append(cnt)
    
    print(f"Found {len(contours)} hold contours (filtered from {len(all_contours)} total contours)")
    
    # Analyze each hold contour for splitting
    splits_needed = []
    
    for i, contour in enumerate(contours):
        should_split, centers = analyze_hold_for_split(contour, mask, video_path)
        
        if should_split:
            print(f"  Contour {i}: Should split into {len(centers)} holds")
            splits_needed.append({
                'contour': contour,
                'centers': centers,
                'index': i
            })
    
    if len(splits_needed) == 0:
        print("\n✅ No grouped holds detected - all holds look good!")
        return
    
    print(f"\n🔍 Found {len(splits_needed)} grouped holds to split")
    
    # Create new holds dictionary
    new_holds = {}
    hold_counter = 0
    
    # Track which original holds we're replacing
    replaced_contours = {s['index'] for s in splits_needed}
    
    # Add holds that don't need splitting
    for hold_id, position in holds.items():
        # Find corresponding contour
        found = False
        for i, contour in enumerate(contours):
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Check if this matches the hold position
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(position))
                if dist < 20:  # Close enough
                    if i not in replaced_contours:
                        # Keep this hold
                        new_holds[f"hold_{hold_counter}"] = position
                        hold_counter += 1
                    found = True
                    break
    
    # Add split holds
    for split_info in splits_needed:
        original_id = f"temp_{split_info['index']}"
        split_holds = split_hold(
            split_info['contour'],
            mask,
            split_info['centers'],
            original_id
        )
        
        for new_hold in split_holds:
            new_holds[f"hold_{hold_counter}"] = new_hold['position']
            hold_counter += 1
    
    # Save new holds JSON
    output_path = Path(output_dir)
    new_holds_path = output_path / "hold_positions_auto_split.json"
    
    with open(new_holds_path, 'w') as f:
        json.dump(new_holds, f, indent=2)
    
    # Create visualization with actual hold contours (NO BOXES)
    vis = clean_frame.copy()
    
    # We need to recreate proper contours for all holds
    # Strategy: Use the mask to find actual contours, then match to hold positions
    
    # Find all contours in the mask
    all_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Match contours to hold positions
    hold_contour_map = {}
    
    for hold_id, position in new_holds.items():
        cx, cy = position
        
        # Find contour that contains or is closest to this point
        best_contour = None
        min_dist = float('inf')
        
        for contour in all_contours:
            # Check if point is inside contour
            dist = cv2.pointPolygonTest(contour, (float(cx), float(cy)), True)
            
            if dist >= -20:  # Inside or very close
                if abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_contour = contour
        
        if best_contour is not None:
            hold_contour_map[hold_id] = best_contour
    
    # Draw all holds with their actual contours
    for hold_id, position in new_holds.items():
        cx, cy = position
        
        # Draw contour (no boxes!)
        if hold_id in hold_contour_map:
            cv2.drawContours(vis, [hold_contour_map[hold_id]], -1, (0, 255, 0), 3)
        
        # Draw center point
        cv2.circle(vis, (cx, cy), 10, (0, 255, 255), -1)
        
        # Draw label
        cv2.putText(vis, hold_id, (cx + 15, cy - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    vis_path = output_path / "holds_debug_split.jpg"
    cv2.imwrite(str(vis_path), vis)
    
    print("\n" + "="*70)
    print(f"✅ SPLIT COMPLETE")
    print("="*70)
    print(f"Original holds: {len(holds)}")
    print(f"Split holds: {len(new_holds)}")
    print(f"Net change: +{len(new_holds) - len(holds)} holds")
    print(f"\nSaved to:")
    print(f"  JSON: {new_holds_path}")
    print(f"  Visualization: {vis_path}")
    print("="*70)
    
    return str(new_holds_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split grouped holds")
    parser.add_argument("--holds", default="output/hold_positions_auto.json", 
                       help="Path to holds JSON")
    parser.add_argument("--mask", default="output/hold_mask_composite.jpg",
                       help="Path to hold mask")
    parser.add_argument("--debug", default="output/holds_debug.jpg",
                       help="Path to debug image")
    parser.add_argument("--output", "-o", default="output",
                       help="Output directory")
    parser.add_argument("--video", "-v", default=None,
                       help="Path to original video (for clean frame)")
    
    args = parser.parse_args()
    
    split_grouped_holds(
        holds_json_path=args.holds,
        mask_path=args.mask,
        debug_image_path=args.debug,
        output_dir=args.output,
        video_path=args.video
    )