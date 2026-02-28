#!/usr/bin/env python3
# analyze_hold_quality.py
#
# Analyze hold quality based on type, size, shape, wall angle, and other features.
# Outputs a quality score from 0.0 (terrible) to 1.0 (perfect jug)

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List
from scipy.spatial import ConvexHull


# Hold type base quality scores (0.0 = hardest, 1.0 = easiest)
HOLD_TYPE_BASE_QUALITY = {
    'jug': 0.95,           # Best holds
    'pocket': 0.75,        # Good but depends on depth
    'pinch': 0.60,         # Moderate difficulty
    'volume': 0.55,        # Depends heavily on angle
    'sloper': 0.35,        # Hard to grip
    'crimp_or_foot': 0.25, # Smallest, hardest
    'unknown': 0.50,       # Neutral
}

# Wall angle adjustments
# Positive angle = overhang (makes holds harder)
# Negative angle = slab (can make some holds easier)
def get_angle_modifier(hold_type: str, wall_angle: float) -> float:
    """
    Calculate quality modifier based on wall angle.
    
    Args:
        hold_type: Type of hold
        wall_angle: Wall angle in degrees (-90 to +90)
                   Negative = slab, 0 = vertical, Positive = overhang
    
    Returns:
        Multiplier for quality (0.5 to 1.2)
    """
    # Normalize angle to 0-1 range
    # -30° (slab) = 0, 0° (vert) = 0.5, +30° (overhang) = 1.0
    angle_factor = (wall_angle + 30) / 60.0
    angle_factor = np.clip(angle_factor, 0.0, 1.0)
    
    # Different holds respond differently to angle
    if hold_type == 'jug':
        # Jugs stay good on any angle
        return 1.0 - (angle_factor * 0.15)  # 1.0 on slab, 0.85 on overhang
    
    elif hold_type == 'sloper':
        # Slopers get MUCH harder on overhang
        return 1.2 - (angle_factor * 0.7)  # 1.2 on slab, 0.5 on overhang
    
    elif hold_type == 'crimp_or_foot':
        # Crimps are bad everywhere, slightly worse on overhang
        return 1.0 - (angle_factor * 0.25)  # 1.0 on slab, 0.75 on overhang
    
    elif hold_type == 'pinch':
        # Pinches are okay on any angle
        return 1.0 - (angle_factor * 0.1)
    
    elif hold_type == 'pocket':
        # Pockets stay consistent
        return 1.0 - (angle_factor * 0.05)
    
    elif hold_type == 'volume':
        # Volumes are highly angle-dependent
        return 1.1 - (angle_factor * 0.5)  # 1.1 on slab, 0.6 on overhang
    
    else:
        # Unknown - assume moderate angle dependence
        return 1.0 - (angle_factor * 0.2)


def analyze_hold_size(contour: np.ndarray, image_height: int) -> Dict[str, float]:
    """
    Analyze hold size features.
    
    Returns dict with:
        - area_pixels: Raw area in pixels
        - area_normalized: Area normalized by image size (0-1)
        - size_quality: Quality score from size (0-1)
    """
    area = cv2.contourArea(contour)
    
    # Normalize by image area
    image_area = image_height * image_height  # Approximate
    area_normalized = area / image_area
    
    # Size quality curve:
    # 0-200px: very small, poor quality (0.3-0.5)
    # 200-800px: small to medium, improving (0.5-0.8)
    # 800-2000px: good size (0.8-1.0)
    # 2000+: very large volumes (0.9-1.0)
    
    if area < 200:
        size_quality = 0.3 + (area / 200) * 0.2  # 0.3 to 0.5
    elif area < 800:
        size_quality = 0.5 + ((area - 200) / 600) * 0.3  # 0.5 to 0.8
    elif area < 2000:
        size_quality = 0.8 + ((area - 800) / 1200) * 0.2  # 0.8 to 1.0
    else:
        size_quality = 0.9 + min(0.1, (area - 2000) / 5000)  # 0.9 to 1.0
    
    return {
        'area_pixels': float(area),
        'area_normalized': float(area_normalized),
        'size_quality': float(size_quality)
    }


def analyze_hold_shape(contour: np.ndarray) -> Dict[str, float]:
    """
    Analyze hold shape features:
    - Solidity: how "filled" the shape is (concave vs convex)
    - Compactness: how circular/regular the shape is
    - Aspect ratio: width/height ratio
    
    Returns quality modifiers based on shape.
    """
    area = cv2.contourArea(contour)
    
    # Convex hull for solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Perimeter for compactness
    perimeter = cv2.arcLength(contour, True)
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    # Bounding box for aspect ratio
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    if width > 0 and height > 0:
        aspect_ratio = max(width, height) / min(width, height)
    else:
        aspect_ratio = 1.0
    
    # Shape quality scoring:
    
    # Solidity (concavity):
    # High solidity (0.9-1.0) = convex, easier to grip (better)
    # Low solidity (0.5-0.7) = concave, might be pocket (could be better)
    # Very low (<0.5) = very irregular, uncertain
    if solidity > 0.85:
        solidity_quality = 0.9  # Convex, good grip
    elif solidity > 0.70:
        solidity_quality = 1.0  # Slightly concave, best (pockets)
    elif solidity > 0.50:
        solidity_quality = 0.8  # Moderate concavity
    else:
        solidity_quality = 0.6  # Very irregular
    
    # Compactness:
    # High (0.8-1.0) = circular, often good holds
    # Medium (0.5-0.8) = elongated, okay
    # Low (<0.5) = very irregular
    compactness_quality = 0.6 + (compactness * 0.4)  # 0.6 to 1.0
    
    # Aspect ratio:
    # 1.0-1.5: compact, good
    # 1.5-3.0: elongated, okay
    # >3.0: very thin, harder
    if aspect_ratio < 1.5:
        aspect_quality = 1.0
    elif aspect_ratio < 3.0:
        aspect_quality = 0.9 - ((aspect_ratio - 1.5) / 1.5) * 0.2  # 0.9 to 0.7
    else:
        aspect_quality = 0.7 - min(0.3, (aspect_ratio - 3.0) / 5.0)  # 0.7 to 0.4
    
    return {
        'solidity': float(solidity),
        'compactness': float(compactness),
        'aspect_ratio': float(aspect_ratio),
        'solidity_quality': float(solidity_quality),
        'compactness_quality': float(compactness_quality),
        'aspect_quality': float(aspect_quality),
        'shape_quality': float(np.mean([solidity_quality, compactness_quality, aspect_quality]))
    }


def analyze_hold_texture(
    image: np.ndarray,
    mask: np.ndarray,
    contour: np.ndarray
) -> Dict[str, float]:
    """
    Analyze hold texture (roughness, color variation).
    More variation might indicate chalk, texture, or features.
    """
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract hold region
    hold_region = image[y:y+h, x:x+w]
    hold_mask = mask[y:y+h, x:x+w]
    
    if hold_region.size == 0:
        return {
            'texture_variance': 0.0,
            'texture_quality': 0.5
        }
    
    # Get pixels inside hold
    hold_pixels = hold_region[hold_mask > 0]
    
    if len(hold_pixels) == 0:
        return {
            'texture_variance': 0.0,
            'texture_quality': 0.5
        }
    
    # Convert to grayscale
    gray_pixels = cv2.cvtColor(hold_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
    
    # Calculate variance
    texture_variance = float(np.var(gray_pixels))
    
    # Texture quality:
    # Higher variance might indicate texture/features (good)
    # But too much might indicate complex lighting (uncertain)
    # Normalize variance (typically 0-2000 range)
    variance_normalized = texture_variance / 2000.0
    
    if variance_normalized < 0.2:
        texture_quality = 0.7  # Very smooth, might be slippery
    elif variance_normalized < 0.6:
        texture_quality = 0.9  # Good texture
    else:
        texture_quality = 0.8  # High variance, uncertain
    
    return {
        'texture_variance': float(texture_variance),
        'texture_quality': float(texture_quality)
    }


def calculate_hold_quality(
    hold_type: str,
    confidence: float,
    size_analysis: Dict,
    shape_analysis: Dict,
    texture_analysis: Dict,
    wall_angle: float,
    position_height: float,  # 0.0 = bottom, 1.0 = top
) -> Dict[str, float]:
    """
    Calculate overall hold quality score.
    
    Returns:
        - quality_score: Overall quality (0.0 = terrible, 1.0 = perfect jug)
        - breakdown: Individual component scores
    """
    # Base quality from hold type
    base_quality = HOLD_TYPE_BASE_QUALITY.get(hold_type, 0.5)
    
    # Confidence modifier: low confidence = more uncertain
    confidence_mod = 0.8 + (confidence * 0.2)  # 0.8 to 1.0
    
    # Wall angle modifier
    angle_mod = get_angle_modifier(hold_type, wall_angle)
    
    # Size quality
    size_qual = size_analysis['size_quality']
    
    # Shape quality
    shape_qual = shape_analysis['shape_quality']
    
    # Texture quality
    texture_qual = texture_analysis['texture_quality']
    
    # Position modifier: higher holds can be harder to reach
    # But this is more about difficulty than quality, so minimal effect
    position_mod = 1.0 - (position_height * 0.05)  # 1.0 at bottom, 0.95 at top
    
    # Weighted combination
    quality = (
        base_quality * 0.40 +      # Type is most important
        size_qual * 0.25 +          # Size matters a lot
        shape_qual * 0.20 +         # Shape is important
        texture_qual * 0.10 +       # Texture helps
        0.05                        # Baseline
    )
    
    # Apply modifiers
    quality *= confidence_mod
    quality *= angle_mod
    quality *= position_mod
    
    # Clamp to 0-1
    quality = np.clip(quality, 0.0, 1.0)
    
    return {
        'quality_score': float(quality),
        'base_quality': float(base_quality),
        'confidence_modifier': float(confidence_mod),
        'angle_modifier': float(angle_mod),
        'size_modifier': float(size_qual),
        'shape_modifier': float(shape_qual),
        'texture_modifier': float(texture_qual),
        'position_modifier': float(position_mod),
        'quality_grade': get_quality_grade(quality)
    }


def get_quality_grade(quality: float) -> str:
    """Convert quality score to letter grade."""
    if quality >= 0.90:
        return 'A+'  # Perfect jugs
    elif quality >= 0.80:
        return 'A'   # Excellent holds
    elif quality >= 0.70:
        return 'B'   # Good holds
    elif quality >= 0.60:
        return 'C'   # Okay holds
    elif quality >= 0.50:
        return 'D'   # Mediocre holds
    else:
        return 'F'   # Bad holds


def analyze_all_holds(
    enriched_holds_path: str,
    mask_path: str,
    image_path: str,
    wall_angle: float = 0.0,
    output_path: str = None
) -> Dict:
    """
    Analyze quality of all holds in a climb.
    
    Args:
        enriched_holds_path: Path to hold_positions_enriched.json
        mask_path: Path to hold_mask_composite.jpg
        image_path: Path to reference image
        wall_angle: Wall angle in degrees
        output_path: Optional output path for enriched JSON
    
    Returns:
        Dictionary with hold quality analysis
    """
    print("\n" + "="*70)
    print("HOLD QUALITY ANALYSIS")
    print("="*70)
    print(f"Wall angle: {wall_angle:.1f}°")
    
    # Load data
    with open(enriched_holds_path, 'r') as f:
        holds = json.load(f)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    
    if mask is None or image is None:
        raise FileNotFoundError("Could not load mask or image")
    
    h, w = image.shape[:2]
    
    # Find all contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Match contours to holds
    hold_contour_map = {}
    for hold_id, hold_data in holds.items():
        cx, cy = hold_data['center']
        
        # Find contour containing this point
        best_contour = None
        min_dist = float('inf')
        
        for contour in contours:
            dist = cv2.pointPolygonTest(contour, (float(cx), float(cy)), True)
            if dist >= -20:  # Inside or very close
                if abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_contour = contour
        
        if best_contour is not None:
            hold_contour_map[hold_id] = best_contour
    
    # Analyze each hold
    results = {}
    
    for hold_id, hold_data in holds.items():
        print(f"\nAnalyzing {hold_id}...")
        
        hold_type = hold_data.get('class', 'unknown')
        confidence = hold_data.get('confidence', 0.5)
        cx, cy = hold_data['center']
        
        # Position (0 = bottom, 1 = top)
        position_height = 1.0 - (cy / h)
        
        # Get contour
        contour = hold_contour_map.get(hold_id)
        
        if contour is None:
            # No contour found, use minimal analysis
            results[hold_id] = {
                **hold_data,
                'quality_analysis': {
                    'quality_score': 0.5,
                    'quality_grade': 'C',
                    'note': 'No contour found for detailed analysis'
                }
            }
            continue
        
        # Run all analyses
        size_analysis = analyze_hold_size(contour, h)
        shape_analysis = analyze_hold_shape(contour)
        texture_analysis = analyze_hold_texture(image, mask, contour)
        
        quality = calculate_hold_quality(
            hold_type=hold_type,
            confidence=confidence,
            size_analysis=size_analysis,
            shape_analysis=shape_analysis,
            texture_analysis=texture_analysis,
            wall_angle=wall_angle,
            position_height=position_height
        )
        
        # Combine all data
        results[hold_id] = {
            **hold_data,
            'size_analysis': size_analysis,
            'shape_analysis': shape_analysis,
            'texture_analysis': texture_analysis,
            'quality_analysis': quality
        }
        
        print(f"  Type: {hold_type} (conf: {confidence:.2f})")
        print(f"  Size: {size_analysis['area_pixels']:.0f}px (quality: {size_analysis['size_quality']:.2f})")
        print(f"  Shape: solidity={shape_analysis['solidity']:.2f}, quality={shape_analysis['shape_quality']:.2f}")
        print(f"  Quality: {quality['quality_score']:.2f} ({quality['quality_grade']})")
    
    # Save enriched results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Saved quality analysis to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("QUALITY SUMMARY")
    print("="*70)
    
    qualities = [r['quality_analysis']['quality_score'] for r in results.values()]
    grades = [r['quality_analysis']['quality_grade'] for r in results.values()]
    
    print(f"Total holds: {len(results)}")
    print(f"Average quality: {np.mean(qualities):.2f}")
    print(f"Min quality: {np.min(qualities):.2f}")
    print(f"Max quality: {np.max(qualities):.2f}")
    print(f"\nGrade distribution:")
    
    from collections import Counter
    grade_counts = Counter(grades)
    for grade in ['A+', 'A', 'B', 'C', 'D', 'F']:
        if grade in grade_counts:
            print(f"  {grade}: {grade_counts[grade]} holds")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze hold quality")
    parser.add_argument("--holds", required=True,
                       help="Path to hold_positions_enriched.json")
    parser.add_argument("--mask", required=True,
                       help="Path to hold_mask_composite.jpg")
    parser.add_argument("--image", required=True,
                       help="Path to reference image")
    parser.add_argument("--angle", type=float, default=0.0,
                       help="Wall angle in degrees")
    parser.add_argument("--output", default=None,
                       help="Output path for quality analysis")
    
    args = parser.parse_args()
    
    if args.output is None:
        # Default: add _quality to the input filename
        base = Path(args.holds).stem
        args.output = str(Path(args.holds).parent / f"{base}_quality.json")
    
    analyze_all_holds(
        enriched_holds_path=args.holds,
        mask_path=args.mask,
        image_path=args.image,
        wall_angle=args.angle,
        output_path=args.output
    )