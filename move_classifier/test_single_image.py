# move_classifier/test_single_image.py

from pathlib import Path
import argparse
import numpy as np
import cv2
import mediapipe as mp
from tensorflow import keras

from move_classifier.move_labels import INDEX_TO_LABEL

# Path to your trained model
MODEL_PATH = Path(__file__).resolve().parent / "saved_models" / "pose_move_classifier.keras"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    print(f"Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    return model


def extract_pose_features(image_bgr, pose) -> np.ndarray | None:
    """
    Run MediaPipe Pose on an image and return a flattened feature vector
    with (x, y, visibility) for all 33 landmarks, or None if no pose.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    coords = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks], dtype=np.float32)
    features = coords.flatten()  # shape (99,)
    return features


def predict_move(image_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # Set up MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    features = extract_pose_features(image_bgr, pose)
    pose.close()

    if features is None:
        print("No pose detected in this image.")
        return

    # Load model
    model = load_model()

    # Predict
    x = features.reshape(1, -1)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = INDEX_TO_LABEL.get(pred_idx, f"unknown_{pred_idx}")

    print(f"\nImage: {image_path}")
    print(f"Predicted move: {pred_label}")
    print("\nClass probabilities:")
    for idx, p in enumerate(probs):
        label = INDEX_TO_LABEL.get(idx, f"unknown_{idx}")
        print(f"  {idx}: {label:10s}  {p:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Predict climbing move from a single image.")
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to the input image file.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    predict_move(image_path)


if __name__ == "__main__":
    main()
