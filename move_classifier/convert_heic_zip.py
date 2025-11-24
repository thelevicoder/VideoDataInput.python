from pathlib import Path
from pillow_heif import register_heif_opener
from PIL import Image
import zipfile
import shutil

register_heif_opener()

# -------------------------
# CONFIGURE THESE PATHS
# -------------------------

INPUT_FOLDER = Path("move_classifier/heic_input")
OUTPUT_FOLDER = Path("move_classifier/move_dataset/Heel_Hook")


def ensure_output_path(stem: str, ext: str = ".jpg") -> Path:
    """Create a non-colliding output path like: name.jpg, name_1.jpg, etc."""
    candidate = OUTPUT_FOLDER / f"{stem}{ext}"
    idx = 1
    while candidate.exists():
        candidate = OUTPUT_FOLDER / f"{stem}_{idx}{ext}"
        idx += 1
    return candidate


def convert_heic_file(heic_path: Path):
    """Convert a single HEIC file to JPG."""
    try:
        img = Image.open(heic_path)
        out_path = ensure_output_path(heic_path.stem)
        img.save(out_path, "JPEG")
        print(f"Converted: {heic_path} -> {out_path}")
    except Exception as e:
        print(f"Error converting {heic_path}: {e}")


def convert_loose_heic_files():
    """Convert any .heic files directly inside INPUT_FOLDER."""
    heic_files = list(INPUT_FOLDER.glob("*.heic")) + list(INPUT_FOLDER.glob("*.HEIC"))

    if not heic_files:
        print("No loose HEIC files found.")
        return

    print(f"Found {len(heic_files)} loose HEIC files.")
    for heic in heic_files:
        convert_heic_file(heic)


def process_zip_file(zip_path: Path):
    """Extract a ZIP, convert HEIC files inside, clean up afterward."""
    extract_root = INPUT_FOLDER / f"_extracted_{zip_path.stem}"
    print(f"Processing zip: {zip_path.name}")
    print(f"Extracting to: {extract_root}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return

    heic_files = list(extract_root.rglob("*.heic")) + list(extract_root.rglob("*.HEIC"))

    if not heic_files:
        print(f"No HEIC files found inside {zip_path.name}")
    else:
        print(f"Found {len(heic_files)} HEIC files inside {zip_path.name}")
        for heic in heic_files:
            convert_heic_file(heic)

    try:
        shutil.rmtree(extract_root)
        print(f"Removed temporary folder: {extract_root}")
    except Exception as e:
        print(f"Error removing temp folder {extract_root}: {e}")


def convert_all():
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder does not exist: {INPUT_FOLDER}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("INPUT_FOLDER:", INPUT_FOLDER.resolve())
    print("OUTPUT_FOLDER:", OUTPUT_FOLDER.resolve())

    convert_loose_heic_files()

    zip_files = list(INPUT_FOLDER.glob("*.zip"))
    if not zip_files:
        print("No zip files found.")
    else:
        print(f"Found {len(zip_files)} zip files.")
        for z in zip_files:
            process_zip_file(z)

    print("Done converting all HEIC files.")


if __name__ == "__main__":
    convert_all()
