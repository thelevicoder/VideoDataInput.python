from pathlib import Path
from pillow_heif import register_heif_opener
from PIL import Image
import shutil

# Enable HEIC reading
register_heif_opener()

# -------------------------
# CONFIGURE THESE PATHS
# -------------------------

# Folder that contains your .HEIC files
INPUT_FOLDER = Path("move_classifier/heic_input")

# Folder where JPGs will be saved
OUTPUT_FOLDER = Path("move_classifier/move_dataset/Toe_Hook")


def convert_heic_to_jpg(input_folder: Path, output_folder: Path):
    heic_files = list(input_folder.glob("*.heic"))

    if not heic_files:
        print("No HEIC files found in:", input_folder)
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(heic_files)} HEIC files.")
    print(f"Saving JPGs to: {output_folder}")

    for heic_file in heic_files:
        try:
            img = Image.open(heic_file)
            jpg_path = output_folder / (heic_file.stem + ".jpg")

            img.save(jpg_path, "JPEG")
            print(f"Converted: {heic_file.name} to {jpg_path.name}")

        except Exception as e:
            print(f"ERROR converting {heic_file}: {e}")


if __name__ == "__main__":
    convert_heic_to_jpg(INPUT_FOLDER, OUTPUT_FOLDER)
