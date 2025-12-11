# enrich_holds_with_yolo.py
#
# Use your trained YOLO hold detector to add type, confidence
# and approximate size to each contour in hold_positions_auto.json,
# by classifying a small crop around each contour center and enforcing
# some geometry sanity checks.

import json
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
from ultralytics import YOLO


def load_contours(json_path: str) -> Dict[str, List[float]]:
    with open(json_path, "r", encoding="utf8") as f:
        return json.load(f)


def classify_contours_with_yolo(
    image_path: str,
    contours: Dict[str, List[int]],
    model_path: str,
    crop_half_size: int = 96,
    conf_thres: float = 0.6,
) -> Dict[str, Dict[str, Any]]:
    """
    For each contour center, crop a square patch around it and run YOLO
    on that patch. Use the best detection that:
      - has conf >= conf_thres
      - contains the contour center
      - has reasonable area relative to the crop
    Otherwise, mark as unknown.
    """
    model = YOLO(model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    enriched: Dict[str, Dict[str, Any]] = {}

    for cid, (cx, cy) in contours.items():
        cx = int(cx)
        cy = int(cy)

        x1 = max(0, cx - crop_half_size)
        y1 = max(0, cy - crop_half_size)
        x2 = min(w, cx + crop_half_size)
        y2 = min(h, cy + crop_half_size)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            enriched[cid] = {
                "center": [cx, cy],
                "class": "unknown",
                "confidence": 0.0,
                "bbox": None,
                "size_px": None,
                "crop_box": [x1, y1, x2, y2],
            }
            continue

        crop_h, crop_w = crop.shape[:2]
        crop_area = float(crop_w * crop_h)

        results = model(crop, imgsz=320, conf=conf_thres, verbose=False)[0]
        boxes = results.boxes

        valid_det = None
        best_conf = -1.0
        names = results.names

        if boxes is not None and len(boxes) > 0:
            # contour center in crop coordinates
            ccx = cx - x1
            ccy = cy - y1

            for box in boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # skip low conf (extra guard even though model already filters)
                if conf < conf_thres:
                    continue

                # does bbox contain contour center?
                if not (bx1 <= ccx <= bx2 and by1 <= ccy <= by2):
                    continue

                # area sanity check
                box_area = max(0.0, (bx2 - bx1) * (by2 - by1))
                if box_area < 0.02 * crop_area:
                    # tiny speck
                    continue
                if box_area > 0.9 * crop_area:
                    # almost whole crop
                    continue

                # keep the highest confidence valid detection
                if conf > best_conf:
                    best_conf = conf
                    valid_det = {
                        "bx1": bx1,
                        "by1": by1,
                        "bx2": bx2,
                        "by2": by2,
                        "conf": conf,
                        "cls_id": cls_id,
                    }

        if valid_det is None:
            enriched[cid] = {
                "center": [cx, cy],
                "class": "unknown",
                "confidence": 0.0,
                "bbox": None,
                "size_px": None,
                "crop_box": [x1, y1, x2, y2],
            }
        else:
            bx1 = valid_det["bx1"]
            by1 = valid_det["by1"]
            bx2 = valid_det["bx2"]
            by2 = valid_det["by2"]
            conf = valid_det["conf"]
            cls_id = valid_det["cls_id"]
            cls_name = names[cls_id]

            # map crop-local bbox back to full-image coords
            gx1 = float(bx1 + x1)
            gy1 = float(by1 + y1)
            gx2 = float(bx2 + x1)
            gy2 = float(by2 + y1)
            size_px = max(0.0, (gx2 - gx1) * (gy2 - gy1))

            enriched[cid] = {
                "center": [cx, cy],
                "class": cls_name,
                "confidence": conf,
                "bbox": [gx1, gy1, gx2, gy2],
                "size_px": size_px,
                "crop_box": [x1, y1, x2, y2],
            }

    return enriched


def draw_debug_overlay(
    image_path: str,
    contours: Dict[str, List[int]],
    enriched: Dict[str, Dict[str, Any]],
    out_path: str,
):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[enrich] Could not read image {image_path}")
        return

    for cid, data in enriched.items():
        cx, cy = data["center"]
        cls_name = data["class"]
        conf = data["confidence"]
        bbox = data.get("bbox")

        # contour center
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 255, 255), -1)
        cv2.putText(
            img,
            f"{cid}:{cls_name}",
            (int(cx) + 4, int(cy) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # draw box only for known classes
        if bbox is not None and cls_name != "unknown":
            x1, y1, x2, y2 = bbox
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                f"{cls_name} {conf:.2f}",
                (int(x1), int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"[enrich] Debug overlay saved to {out_path}")


def main():
    project_root = Path(".")
    holds_json = project_root / "output" / "hold_positions_auto.json"
    frame_image = project_root / "output" / "holds_debug.jpg"

    # point this at the run that trained well
    model_path = project_root / "runs" / "detect" / "train" / "weights" / "best.pt"

    if not holds_json.exists():
        raise FileNotFoundError(f"Holds JSON not found at {holds_json}")
    if not frame_image.exists():
        raise FileNotFoundError(f"Frame image not found at {frame_image}")
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found at {model_path}")

    contours = load_contours(str(holds_json))
    enriched = classify_contours_with_yolo(
        str(frame_image),
        contours,
        str(model_path),
        crop_half_size=96,
        conf_thres=0.6,
    )

    out_json = project_root / "output" / "hold_positions_enriched.json"
    with out_json.open("w", encoding="utf8") as f:
        json.dump(enriched, f, indent=2)
    print(f"[enrich] Saved enriched holds JSON to {out_json}")

    debug_out = project_root / "output" / "holds_enriched_overlay.jpg"
    draw_debug_overlay(str(frame_image), contours, enriched, str(debug_out))


if __name__ == "__main__":
    main()
