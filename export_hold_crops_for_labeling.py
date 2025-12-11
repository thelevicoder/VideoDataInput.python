# export_hold_crops_for_labeling.py
#
# Take holds_debug.jpg + hold_positions_auto.json and export
# one small image crop per contour so you can label them in Roboflow.

import json
from pathlib import Path

import cv2


def main():
    project_root = Path(".")
    holds_json = project_root / "output" / "hold_positions_auto.json"
    frame_image = project_root / "output" / "holds_debug.jpg"
    out_dir = project_root / "label_crops"

    if not holds_json.exists():
        raise FileNotFoundError(f"Holds JSON not found at {holds_json}")
    if not frame_image.exists():
        raise FileNotFoundError(f"Frame image not found at {frame_image}")

    with holds_json.open("r", encoding="utf8") as f:
        contours = json.load(f)

    img = cv2.imread(str(frame_image))
    if img is None:
        raise FileNotFoundError(f"Could not read image {frame_image}")

    h, w = img.shape[:2]
    crop_half_size = 96

    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for cid, (cx, cy) in contours.items():
        cx = int(cx)
        cy = int(cy)

        x1 = max(0, cx - crop_half_size)
        y1 = max(0, cy - crop_half_size)
        x2 = min(w, cx + crop_half_size)
        y2 = min(h, cy + crop_half_size)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"[export] Empty crop for {cid}, skipping")
            continue

        out_path = out_dir / f"{cid}.jpg"
        cv2.imwrite(str(out_path), crop)
        count += 1

    print(f"[export] Wrote {count} crops to {out_dir}")


if __name__ == "__main__":
    main()
