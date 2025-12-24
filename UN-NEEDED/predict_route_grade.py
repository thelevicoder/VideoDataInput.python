# predict_route_grade.py
#
# Predict climbing route grade from a wall photo and hold color.
#
# Usage:
#   python predict_route_grade.py --image wall.jpg --color "0,255,255"

from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import joblib


def load_model():
    """Load the trained grade prediction model."""
    models_dir = Path('models')
    
    model_path = models_dir / 'grade_predictor.pkl'
    encoder_path = models_dir / 'grade_label_encoder.pkl'
    metadata_path = models_dir / 'grade_predictor_metadata.json'
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Train a model first with: python train_grade_predictor.py"
        )
    
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    
    with metadata_path.open('r') as f:
        metadata = json.load(f)
    
    print(f"✅ Loaded grade predictor")
    print(f"   Trained on: {metadata['grades']}")
    
    return model, label_encoder, metadata


def detect_holds_from_image(image_path: str, target_color_bgr: tuple):
    """
    Simplified hold detection for inference.
    Extract holds of the target color from a static image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Convert target BGR to LAB and HSV
    target_bgr_arr = np.uint8([[target_color_bgr]])
    target_lab = cv2.cvtColor(target_bgr_arr, cv2.COLOR_BGR2LAB)[0][0]
    target_hsv = cv2.cvtColor(target_bgr_arr, cv2.COLOR_BGR2HSV)[0][0]
    
    # Normalize LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Color matching
    LAB_TOL = 18
    HSV_TOL = (12, 40, 40)
    
    diff_lab = np.linalg.norm(lab - target_lab, axis=2)
    mask_lab = (diff_lab < LAB_TOL).astype(np.uint8) * 255
    
    hue_diff = np.abs(hsv[:, :, 0] - target_hsv[0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    diff_s = np.abs(hsv[:, :, 1] - target_hsv[1])
    diff_v = np.abs(hsv[:, :, 2] - target_hsv[2])
    
    mask_hsv = (
        (hue_diff < HSV_TOL[0]) &
        (diff_s < HSV_TOL[1]) &
        (diff_v < HSV_TOL[2])
    ).astype(np.uint8) * 255
    
    mask = cv2.bitwise_and(mask_lab, mask_hsv)
    
    # Morphology
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and extract hold centers
    holds = {}
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if 200 < area < 8000:  # reasonable hold size
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                holds[f"hold_{idx}"] = {
                    "center": [cx, cy],
                    "class": "unknown",  # Would need classifier
                    "confidence": 0.0
                }
    
    print(f"✅ Detected {len(holds)} holds")
    return holds, img


def estimate_features_simple(holds: dict, climber_height: float = 70, climber_wingspan: float = 70, wall_angle: float = 0):
    """
    Estimate route features from detected holds.
    This is a simplified version for inference without move data.
    """
    
    if not holds:
        raise ValueError("No holds detected!")
    
    total_holds = len(holds)
    
    # Get spatial extent
    positions = [h['center'] for h in holds.values()]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    
    route_width = max(xs) - min(xs)
    route_height = max(ys) - min(ys)
    hold_density = total_holds / (route_width * route_height) if route_width * route_height > 0 else 0
    
    # Estimate distances between consecutive holds (sorted by height)
    sorted_positions = sorted(positions, key=lambda p: -p[1])  # top to bottom
    distances = []
    for i in range(len(sorted_positions) - 1):
        p1 = sorted_positions[i]
        p2 = sorted_positions[i + 1]
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        distances.append(dist)
    
    body_unit = climber_height
    normalized_distances = [d / body_unit for d in distances] if distances else [0]
    
    # Since we don't have actual move data or hold classifications,
    # we'll use conservative estimates
    features = {
        'total_holds': total_holds,
        'total_moves': max(total_holds - 1, 1),  # estimate
        
        # Unknown hold distribution - use defaults
        'pct_crimp_or_foot': 0.2,
        'pct_jug': 0.15,
        'pct_pinch': 0.1,
        'pct_pocket': 0.05,
        'pct_sloper': 0.2,
        'pct_volume': 0.25,
        'pct_unknown': 0.05,
        
        # Unknown move distribution - use defaults
        'pct_reach': 0.8,
        'pct_heel_hook': 0.05,
        'pct_toe_hook': 0.05,
        'pct_flag': 0.05,
        'pct_smear': 0.03,
        'pct_figure_4': 0.01,
        'pct_bat_hang': 0.01,
        
        # Distance metrics from spatial analysis
        'avg_move_distance': float(np.mean(normalized_distances)),
        'max_move_distance': float(np.max(normalized_distances)),
        'min_move_distance': float(np.min(normalized_distances)),
        'std_move_distance': float(np.std(normalized_distances)),
        
        'pct_long_reaches': sum(1 for d in normalized_distances if d > 2.0) / len(normalized_distances) if normalized_distances else 0,
        'max_reach_ratio': float(np.max(normalized_distances)) / (climber_wingspan / body_unit) if climber_wingspan > 0 else 0,
        
        'route_height': route_height,
        'route_width': route_width,
        'hold_density': hold_density,
        
        'climber_height_inches': climber_height,
        'climber_wingspan_inches': climber_wingspan,
        'wall_angle': wall_angle,
    }
    
    return features


def predict_grade(model, label_encoder, features_dict, metadata):
    """Predict route grade from features."""
    
    # Convert features dict to array in correct order
    feature_names = metadata['feature_names']
    feature_vector = [features_dict[name] for name in feature_names]
    
    X = pd.DataFrame([feature_vector], columns=feature_names)
    
    # Predict
    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0]
    
    predicted_grade = label_encoder.inverse_transform([y_pred])[0]
    confidence = y_proba[y_pred]
    
    # Get top 3 predictions
    top_3_indices = np.argsort(y_proba)[-3:][::-1]
    top_3_predictions = [
        (label_encoder.inverse_transform([idx])[0], y_proba[idx])
        for idx in top_3_indices
    ]
    
    return predicted_grade, confidence, top_3_predictions


def visualize_route(img, holds, predicted_grade, confidence):
    """Draw holds on image with prediction."""
    vis = img.copy()
    
    for hold_id, hold_data in holds.items():
        cx, cy = hold_data['center']
        cv2.circle(vis, (cx, cy), 8, (0, 255, 255), -1)
        cv2.putText(vis, hold_id, (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Draw prediction
    text = f"Predicted Grade: {predicted_grade} ({confidence*100:.1f}%)"
    cv2.putText(vis, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Predict climbing route grade from wall photo"
    )
    parser.add_argument('--image', '-i', required=True, help='Path to wall image')
    parser.add_argument(
        '--color', '-c', required=True,
        help='Hold color in BGR format (e.g., "255,0,0" for blue)'
    )
    parser.add_argument(
        '--height', type=float, default=70,
        help='Climber height in inches (default: 70)'
    )
    parser.add_argument(
        '--wingspan', type=float, default=70,
        help='Climber wingspan in inches (default: 70)'
    )
    parser.add_argument(
        '--angle', type=float, default=0,
        help='Wall angle in degrees (default: 0=vertical)'
    )
    parser.add_argument(
        '--output', '-o', default='output/predicted_route.jpg',
        help='Path to save visualization'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CLIMBING ROUTE GRADE PREDICTION")
    print("="*60 + "\n")
    
    # Parse color
    try:
        color_parts = [int(x.strip()) for x in args.color.split(',')]
        if len(color_parts) != 3:
            raise ValueError()
        target_color = tuple(color_parts)
    except:
        print("❌ Error: Color must be in BGR format (e.g., '255,0,0')")
        return
    
    # Load model
    try:
        model, label_encoder, metadata = load_model()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Detect holds
    print(f"\nDetecting holds of color BGR{target_color} in {args.image}...")
    holds, img = detect_holds_from_image(args.image, target_color)
    
    if not holds:
        print("❌ No holds detected! Try adjusting the color or image.")
        return
    
    # Extract features
    print("\nExtracting route features...")
    print(f"  Climber height: {args.height} inches")
    print(f"  Climber wingspan: {args.wingspan} inches")
    print(f"  Wall angle: {args.angle}°")
    
    features = estimate_features_simple(holds, args.height, args.wingspan, args.angle)
    
    # Predict
    print("\nPredicting grade...")
    predicted_grade, confidence, top_3 = predict_grade(
        model, label_encoder, features, metadata
    )
    
    # Results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\n🎯 Predicted Grade: {predicted_grade}")
    print(f"   Confidence: {confidence*100:.1f}%")
    print(f"\nTop 3 Predictions:")
    for grade, prob in top_3:
        print(f"  {grade}: {prob*100:.1f}%")
    
    print(f"\n📊 Route Statistics:")
    print(f"  Total Holds: {features['total_holds']}")
    print(f"  Avg Move Distance: {features['avg_move_distance']:.2f} body units")
    print(f"  Max Reach: {features['max_move_distance']:.2f} body units")
    print(f"  Route Height: {features['route_height']:.0f} pixels")
    
    if confidence < 0.4:
        print("\n⚠️  LOW CONFIDENCE WARNING")
        print("   This prediction may not be reliable.")
        print("   The route characteristics may be unusual or outside training data.")
    
    print("="*60 + "\n")
    
    # Visualize
    vis = visualize_route(img, holds, predicted_grade, confidence)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)
    print(f"✅ Saved visualization to {output_path}\n")


if __name__ == "__main__":
    main()
    