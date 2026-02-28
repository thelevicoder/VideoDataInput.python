#!/usr/bin/env python3
# split_grouped_holds_improved.py
#
# IMPROVED: More conservative splitting with better heuristics
# Only splits holds that are CLEARLY multiple holds grouped together

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial import distance
from sklearn.cluster import DBSCAN


def should_split_hold(contour: np.ndarray, mask: np.ndarray) -> bool:
    """
    Determine if a hold should be split based on multiple factors.
    
    More conservative approach - only split when confident it's multiple holds.
    """
    area = cv2.contourArea(contour)
    
    # Don't split small holds (< 500px)
    if area < 500:
        return False
    
    # Calculate aspect ratio
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    if width == 0 or height == 0:
        return False
    
    aspect_ratio = max(width, height) / min(width, height)
    
    # Calculate solidity (how "filled" the shape is)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 1.0
    
    # Calculate compactness (how circular)
    perimeter = cv2.arcLength(contour, True)
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    extent = area / bbox_area if bbox_area > 0 else 0
    
    # SPLITTING CRITERIA (must meet MULTIPLE conditions):
    
    # 1. Very elongated AND low solidity (likely multiple holds in a line)
    if aspect_ratio > 3.5 and solidity < 0.75:
        return True
    
    # 2. Large, irregular shape with low compactness and solidity
    if area > 2000 and solidity < 0.65 and compactness < 0.4:
        return True
    
    # 3. Very large with multiple clear gaps
    if area > 3000 and solidity < 0.70:
        return True
    
    # 4. Extremely elongated (likely wrongly merged)
    if aspect_ratio > 5.0:
        return True
    
    # Otherwise, DON'T split
    return False


def find_split_candidates_watershed(
    mask: np.ndarray,
    contour: np.ndarray,
    min_separation: int = 40
) -> List[Tuple[int, int]]:
    """
    Use watershed algorithm to find split points.
    More robust than simple erosion.
    """
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract region
    region_mask = np.zeros_like(mask)
    cv2.drawContours(region_mask, [contour], -1, 255, -1)
    region_mask = region_mask[y:y+h, x:x+w]
    
    if region_mask.size == 0:
        return []
    
    # Distance transform
    dist_transform = cv2.distanceTransform(region_mask, cv2.DIST_L2, 5)
    
    # Find local maxima (peaks)
    kernel = np.ones((min_separation//2, min_separation//2), np.uint8)
    local_max = cv2.dilate(dist_transform, kernel)
    local_max = (dist_transform == local_max) & (dist_transform > min_separation/4)
    
    # Get peak coordinates
    peak_coords = np.argwhere(local_max)
    
    if len(peak_coords) < 2:
        return []
    
    # Convert to absolute coordinates
    centers = [(int(pt[1] + x), int(pt[0] + y)) for pt in peak_coords]
    
    # Filter: peaks must be reasonably separated
    filtered_centers = []
    for i, c1 in enumerate(centers):
        too_close = False
        for j, c2 in enumerate(filtered_centers):
            if distance.euclidean(c1, c2) < min_separation:
                too_close = True
                break
        if not too_close:
            filtered_centers.append(c1)
    
    return filtered_centers if len(filtered_centers) >= 2 else []


def find_split_candidates_clustering(
    mask: np.ndarray,
    contour: np.ndarray,
    min_separation: int = 40
) -> List[Tuple[int, int]]:
    """
    Use DBSCAN clustering on contour points to find natural groupings.
    """
    # Get all points in the contour
    points = contour.reshape(-1, 2)
    
    if len(points) < 20:
        return []
    
    # Sample points for efficiency
    if len(points) > 200:
        indices = np.random.choice(len(points), 200, replace=False)
        points = points[indices]
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=min_separation, min_samples=5).fit(points)
    labels = clustering.labels_
    
    # Get centers of each cluster
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Remove noise
    
    if len(unique_labels) < 2:
        return []
    
    centers = []
    for label in unique_labels:
        cluster_points = points[labels == label]
        center = cluster_points.mean(axis=0)
        centers.append((int(center[0]), int(center[1])))
    
    return centers


def split_hold_into_subholds(
    mask: np.ndarray,
    contour: np.ndarray,
    centers: List[Tuple[int, int]]
) -> List[np.ndarray]:
    """
    Split a hold into sub-holds using voronoi-like partitioning.
    """
    if len(centers) < 2:
        return [contour]
    
    # Create a blank mask for each center
    h, w = mask.shape
    submasks = [np.zeros((h, w), dtype=np.uint8) for _ in centers]
    
    # For each pixel in the contour, assign to nearest center
    x, y, cw, ch = cv2.boundingRect(contour)
    
    for py in range(y, y + ch):
        for px in range(x, x + cw):
            if cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0:
                # Find nearest center
                distances = [distance.euclidean((px, py), c) for c in centers]
                nearest = np.argmin(distances)
                submasks[nearest][py, px] = 255
    
    # Find contours in each submask
    subcontours = []
    for submask in submasks:
        contours, _ = cv2.findContours(submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Take the largest contour
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:  # Minimum size
                subcontours.append(largest)
    
    return subcontours if len(subcontours) >= 2 else [contour]


def split_grouped_holds(
    holds_json_path: str,
    mask_path: str,
    output_dir: str,
    min_separation: int = 40,
    debug: bool = True
) -> Dict:
    """
    IMPROVED split detection with conservative heuristics.
    
    Args:
        holds_json_path: Path to hold_positions_auto.json
        mask_path: Path to hold_mask_composite.jpg
        output_dir: Output directory
        min_separation: Minimum separation between hold centers (pixels)
        debug: Whether to save debug images
    
    Returns:
        Dictionary with split hold positions
    """
    print("\n" + "="*70)
    print("IMPROVED HOLD SPLITTING")
    print("="*70)
    
    # Load data
    with open(holds_json_path, 'r') as f:
        original_holds = json.load(f)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")
    
    # Load debug image if available
    output_dir = Path(output_dir)
    debug_image_path = output_dir / "holds_debug.jpg"
    if debug_image_path.exists():
        debug_img = cv2.imread(str(debug_image_path))
    else:
        debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Find all contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} contours in mask")
    print(f"Original holds: {len(original_holds)}")
    print(f"Minimum separation: {min_separation}px")
    
    # Match original holds to contours
    hold_contour_map = {}
    for hold_id, (cx, cy) in original_holds.items():
        best_contour = None
        min_dist = float('inf')
        
        for contour in contours:
            dist = cv2.pointPolygonTest(contour, (float(cx), float(cy)), True)
            if dist >= -20:
                if abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_contour = contour
        
        if best_contour is not None:
            hold_contour_map[hold_id] = best_contour
    
    # Analyze and split holds
    split_holds = {}
    split_count = 0
    kept_count = 0
    
    for hold_id, contour in hold_contour_map.items():
        # Check if this hold should be split
        if not should_split_hold(contour, mask):
            # Keep as-is
            cx, cy = original_holds[hold_id]
            split_holds[hold_id] = [int(cx), int(cy)]
            kept_count += 1
            
            if debug:
                cv2.circle(debug_img, (int(cx), int(cy)), 8, (0, 255, 0), -1)
                cv2.putText(debug_img, hold_id, (int(cx) + 15, int(cy)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            continue
        
        # Try to find split candidates
        centers_watershed = find_split_candidates_watershed(mask, contour, min_separation)
        centers_clustering = find_split_candidates_clustering(mask, contour, min_separation)
        
        # Use the method that found more centers (if any)
        if len(centers_watershed) >= 2 or len(centers_clustering) >= 2:
            if len(centers_watershed) >= len(centers_clustering):
                centers = centers_watershed
                method = "watershed"
            else:
                centers = centers_clustering
                method = "clustering"
            
            print(f"\n{hold_id}: Splitting into {len(centers)} holds ({method})")
            area = cv2.contourArea(contour)
            print(f"  Area: {area:.0f}px")
            
            # Split the hold
            subcontours = split_hold_into_subholds(mask, contour, centers)
            
            if len(subcontours) >= 2:
                # Successfully split
                for i, subcontour in enumerate(subcontours):
                    new_id = f"{hold_id}_{i}"
                    M = cv2.moments(subcontour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        split_holds[new_id] = [cx, cy]
                        split_count += 1
                        
                        if debug:
                            cv2.circle(debug_img, (cx, cy), 8, (0, 255, 255), -1)
                            cv2.putText(debug_img, new_id, (cx + 15, cy),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                # Splitting failed, keep original
                cx, cy = original_holds[hold_id]
                split_holds[hold_id] = [int(cx), int(cy)]
                kept_count += 1
        else:
            # No split candidates found, keep original
            cx, cy = original_holds[hold_id]
            split_holds[hold_id] = [int(cx), int(cy)]
            kept_count += 1
            
            if debug:
                cv2.circle(debug_img, (int(cx), int(cy)), 8, (0, 255, 0), -1)
                cv2.putText(debug_img, hold_id, (int(cx) + 15, int(cy)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save results
    output_json = output_dir / "hold_positions_auto_split.json"
    with open(output_json, 'w') as f:
        json.dump(split_holds, f, indent=2)
    
    if debug:
        debug_output = output_dir / "holds_debug_split_improved.jpg"
        cv2.imwrite(str(debug_output), debug_img)
        print(f"\n✅ Debug image: {debug_output}")
    
    print(f"\n✅ Split results: {output_json}")
    print(f"   Original holds: {len(original_holds)}")
    print(f"   Kept unchanged: {kept_count}")
    print(f"   Split into: {split_count}")
    print(f"   Total holds: {len(split_holds)}")
    print("="*70)
    
    return split_holds


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved hold splitting")
    parser.add_argument("--holds", required=True,
                       help="Path to hold_positions_auto.json")
    parser.add_argument("--mask", required=True,
                       help="Path to hold_mask_composite.jpg")
    parser.add_argument("--output", required=True,
                       help="Output directory")
    parser.add_argument("--min-separation", type=int, default=40,
                       help="Minimum separation between holds (pixels)")
    parser.add_argument("--no-debug", action="store_true",
                       help="Disable debug output")
    
    args = parser.parse_args()
    
    split_grouped_holds(
        holds_json_path=args.holds,
        mask_path=args.mask,
        output_dir=args.output,
        min_separation=args.min_separation,
        debug=not args.no_debug
    )