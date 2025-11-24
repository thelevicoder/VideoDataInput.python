# move_classifier/dataset_builder.py
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp

from .move_labels import FOLDER_TO_LABEL, LABEL_TO_INDEX

# ---- CONFIG ----
# Directory of this file: .../VideoDataInput.python/move_classifier
THIS_DIR = Path(__file__).resolve().parent

# Dataset folder is inside move_classifier/
DATASET_ROOT = THIS_DIR / "move_dataset"

# Save the output .npz next to this script (or change if you prefer)
OUTPUT_PATH = THIS_DIR / "move_pose_dataset.npz"

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images():
    """Yield (image_path, folder_name) for all images in move_dataset."""
    print(f"[DEBUG] Looking for dataset root at: {DATASET_ROOT}")
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")
    
    for class_dir in DATASET_ROOT.iterdir():
        if not class_dir.is_dir():
            continue

        folder_name = class_dir.name  # e.g. "Heel_Hook"
        if folder_name not in FOLDER_TO_LABEL:
            print(f"[WARN] Skipping unknown folder: {folder_name}")
            continue

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTS:
                yield img_path, folder_name


def extract_pose_features(image_path: Path, pose) -> np.ndarray | None:
    """
    Run MediaPipe Pose on an image and return a flattened feature vector.

    For v1: we simply use (x, y, visibility) for all 33 landmarks.
    MediaPipe already normalizes x,y to [0,1], so this is a good start.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[WARN] Failed to read image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"[WARN] No pose detected in: {image_path}")
        return None

    landmarks = results.pose_landmarks.landmark  # list of 33
    # Each landmark: x, y are normalized [0,1]; visibility in [0,1]
    coords = np.array(
        [[lm.x, lm.y, lm.visibility] for lm in landmarks],
        dtype=np.float32,
    )  # shape: (33, 3)

    # Flatten to 1D feature vector (33 * 3 = 99 dims)
    features = coords.flatten()  # shape: (99,)
    return features


def build_dataset():
    print(f"🔍 Scanning dataset at: {DATASET_ROOT.resolve()}")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    paths: list[str] = []

    total = 0
    used = 0

    for img_path, folder_name in iter_images():
        total += 1
        label_str = FOLDER_TO_LABEL[folder_name]      # e.g. "heel_hook"
        label_idx = LABEL_TO_INDEX[label_str]         # e.g. 1

        features = extract_pose_features(img_path, pose)
        if features is None:
            continue

        X_list.append(features)
        y_list.append(label_idx)
        # store relative path for debugging / traceability
        paths.append(str(img_path.relative_to(DATASET_ROOT)))
        used += 1

        if used % 50 == 0:
            print(f"  Processed {used} images with valid pose...")

    pose.close()

    if not X_list:
        raise RuntimeError("No valid samples collected! Check your dataset and pose detection.")

    X = np.stack(X_list, axis=0)          # shape: (N, 99)
    y = np.array(y_list, dtype=np.int64)  # shape: (N,)
    paths_arr = np.array(paths)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_PATH, X=X, y=y, paths=paths_arr)

    print("\n✅ Dataset build complete!")
    print(f"   Total images scanned: {total}")
    print(f"   Samples with pose:    {used}")
    print(f"   Saved to:             {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    build_dataset()
