import os
from pathlib import Path
from PIL import Image, ImageEnhance

# Path to your dataset folders
DATASET_ROOT = Path("move_classifier/move_dataset")

# File suffixes we generate (so we don't re-augment them)
AUG_SUFFIXES = ["_flip", "_rot", "_bright"]

def is_augmented(filename: str) -> bool:
    """Return True if this file has already been augmented."""
    base = filename.lower()
    return any(suffix in base for suffix in AUG_SUFFIXES)

def augment_image(img_path: Path):
    """Create augmented versions of the image."""
    img = Image.open(img_path)

    # 1. Horizontal flip
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped.save(img_path.with_name(img_path.stem + "_flip.jpg"))

    # 2. Small rotation (+10 degrees)
    rotated = img.rotate(10, resample=Image.BICUBIC, expand=False)
    rotated.save(img_path.with_name(img_path.stem + "_rot.jpg"))

    # 3. Increase brightness slightly
    enhancer = ImageEnhance.Brightness(img)
    brighter = enhancer.enhance(1.25)
    brighter.save(img_path.with_name(img_path.stem + "_bright.jpg"))

    print(f"Augmented: {img_path.name}")

def main():
    if not DATASET_ROOT.exists():
        print(f"Dataset folder not found: {DATASET_ROOT}")
        return

    print(f"Augmenting dataset at: {DATASET_ROOT}")

    for class_folder in DATASET_ROOT.iterdir():
        if not class_folder.is_dir():
            continue

        print(f"\nProcessing class folder: {class_folder.name}")

        for img_path in class_folder.iterdir():
            if img_path.suffix.lower() != ".jpg":
                continue

            if is_augmented(img_path.name):
                # Skip files we already created
                continue

            # Create augmented versions
            augment_image(img_path)

    print("\n Augmentation complete! Now run:")
    print("   python -m move_classifier.dataset_builder")
    print("   python -m move_classifier.train_classifier")

if __name__ == "__main__":
    main()
