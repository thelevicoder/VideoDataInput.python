# convert_hold_detection_to_patches.py
#
# Convert YOLO-style detection dataset (full wall images + .txt labels)
# into a classification-style dataset of cropped holds:
#
# dataset/HoldClass/Final_Dataset/patches/train/<class_name>/*.jpg
#
# Then train_hold_classifier.py can use ImageFolder on patches/.

from pathlib import Path
import cv2
import os

# ---------- CONFIG ----------

# Root of the Kaggle dataset relative to your project
DATASET_ROOT = Path("dataset") / "HoldClass" / "Final_Dataset"

# Splits we will convert
SPLITS = ["train", "valid", "test"]

# Minimum crop size in pixels to keep
MIN_W = 10
MIN_H = 10

# Optional: mapping from numeric class id -> human-readable name.
# If you know the exact mapping from the dataset docs, fill it in here.
# Otherwise we fall back to "class_0", "class_1", etc.
CLASS_MAP = {
    0: "class_0",
    1: "class_1",
    2: "class_2",
    3: "class_3",
    4: "class_4",
    5: "class_5",
}

# ----------------------------


def yolo_to_xyxy(rel_cx, rel_cy, rel_w, rel_h, img_w, img_h):
    """Convert relative YOLO bbox to absolute pixel xyxy."""
    cx = rel_cx * img_w
    cy = rel_cy * img_h
    bw = rel_w * img_w
    bh = rel_h * img_h

    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))

    # Clamp to image bounds
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    return x1, y1, x2, y2


def process_split(split: str):
    images_dir = DATASET_ROOT / split / "images"
    labels_dir = DATASET_ROOT / split / "labels"
    out_root = DATASET_ROOT / "patches" / split

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    print(f"[convert] Processing split '{split}'")
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    count_total = 0
    count_kept = 0

    for img_path in image_paths:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[convert] Could not read image {img_path}")
            continue

        h, w = img.shape[:2]

        with open(label_path, "r", encoding="utf8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                cls_id = int(parts[0])
                rel_cx = float(parts[1])
                rel_cy = float(parts[2])
                rel_w = float(parts[3])
                rel_h = float(parts[4])
            except ValueError:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(rel_cx, rel_cy, rel_w, rel_h, w, h)
            crop = img[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]

            count_total += 1

            # Filter out tiny crops
            if cw < MIN_W or ch < MIN_H:
                continue

            cls_name = CLASS_MAP.get(cls_id, f"class_{cls_id}")
            out_dir = out_root / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{stem}_{i}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), crop)
            count_kept += 1

    print(f"[convert] Split '{split}': kept {count_kept} / {count_total} crops")


def main():
    for split in SPLITS:
        process_split(split)

    print("[convert] Done. Classification patches in:")
    print(f"  {DATASET_ROOT / 'patches' / 'train'}")
    print(f"  {DATASET_ROOT / 'patches' / 'valid'}")
    print(f"  {DATASET_ROOT / 'patches' / 'test'}")


if __name__ == "__main__":
    main()
