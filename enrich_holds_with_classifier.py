# enrich_holds_with_classifier.py
#
# Use the trained hold classifier to assign a class and confidence
# to each contour center in output/hold_positions_auto.json.
#
# It also saves a debug crop image for every contour into debug_patches/
# so you can visually check what the classifier is looking at.
#
# Inputs:
#   output/hold_positions_auto.json   (from your hold color detector)
#   output/holds_debug.jpg            (color snapshot with holds)
#   models/hold_classifier_resnet18.pt
#   models/hold_class_labels.json
#
# Outputs:
#   output/hold_positions_enriched.json
#   output/holds_enriched_overlay.jpg
#   debug_patches/patch_<contour_id>.jpg

from pathlib import Path
import json
from typing import Dict, Any, List

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

def is_likely_person_occlusion(crop: np.ndarray) -> bool:
    """
    Very conservative heuristic.
    Only return True if the crop is clearly mostly person
    (large majority skin tones or very dark cloth).
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    total = crop.shape[0] * crop.shape[1]
    if total == 0:
        return False

    # skin-ish tones in HSV (very rough)
    skin_mask = (
        ((h >= 0) & (h <= 25)) & (s > 50) & (v > 50)
    ) | (
        ((h >= 160) & (h <= 180)) & (s > 50) & (v > 50)
    )
    skin_ratio = float(np.count_nonzero(skin_mask)) / float(total)

    # very dark cloth
    dark_mask = (v < 30)
    dark_ratio = float(np.count_nonzero(dark_mask)) / float(total)

    # only treat as occluded if it is *mostly* skin or *mostly* dark
    return (skin_ratio > 0.7) or (dark_ratio > 0.8)



def load_holds(json_path: Path) -> Dict[str, List[float]]:
    with json_path.open("r", encoding="utf8") as f:
        return json.load(f)


def build_model(model_path: Path, labels_path: Path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    with labels_path.open("r", encoding="utf8") as f:
        label_data = json.load(f)
    classes = label_data["classes"]

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return model, classes, tfm


def crop_around_center(img: np.ndarray, cx: float, cy: float, size: int = 160):
    """Return a square crop around (cx, cy) and its bbox [x1, y1, x2, y2]."""
    h, w = img.shape[:2]
    half = size // 2

    cx = int(round(cx))
    cy = int(round(cy))

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = img[y1:y2, x1:x2].copy()
    return crop, [x1, y1, x2, y2]


def classify_crop(
    model,
    tfm,
    crop: np.ndarray,
    device: torch.device,
):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = tfm(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)

    return int(idx.item()), float(conf.item())


def draw_debug_overlay(
    img: np.ndarray,
    holds: Dict[str, List[float]],
    enriched: Dict[str, Dict[str, Any]],
    out_path: Path,
):
    for cid, center in holds.items():
        cx, cy = center
        data = enriched.get(cid, {})
        cls_name = data.get("class", "unknown")
        conf = data.get("confidence", 0.0)
        bbox = data.get("bbox")

        # contour center
        cv2.circle(img, (int(cx), int(cy)), 4, (0, 255, 255), -1)
        cv2.putText(
            img,
            cid,
            (int(cx) + 4, int(cy) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # draw box and label if known
        if bbox is not None:
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
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[enrich] Debug overlay saved to {out_path}")


def main():
    project_root = Path(".")

    holds_json = project_root / "output" / "hold_positions_auto.json"
    frame_image = project_root / "output" / "holds_debug.jpg"
    model_path = project_root / "models" / "hold_classifier_resnet18.pt"
    labels_path = project_root / "models" / "hold_class_labels.json"

    if not holds_json.exists():
        raise FileNotFoundError(f"Holds JSON not found: {holds_json}")
    if not frame_image.exists():
        raise FileNotFoundError(f"Frame image not found: {frame_image}")
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Class labels file not found: {labels_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    holds = load_holds(holds_json)
    model, classes, tfm = build_model(model_path, labels_path, device)
    print("Classifier classes:", classes)

    img = cv2.imread(str(frame_image))
    if img is None:
        raise RuntimeError(f"Could not read frame image {frame_image}")

    # folder for visual debugging of crops
    debug_dir = project_root / "debug_patches"
    debug_dir.mkdir(exist_ok=True)

    enriched: Dict[str, Dict[str, Any]] = {}

    for cid, center in holds.items():
        cx, cy = center
        crop, bbox = crop_around_center(img, cx, cy, size=160)
        if crop is None:
            enriched[cid] = {
                "center": center,
                "class": "unknown",
                "confidence": 0.0,
                "bbox": None,
            }
            continue

        # save the raw crop so we can see what the model sees
                # save the raw crop so we can see what the model sees
        debug_path = debug_dir / f"patch_{cid}.jpg"
        cv2.imwrite(str(debug_path), crop)

        # if the crop looks mostly like person, do not trust classification
        if is_likely_person_occlusion(crop):
            enriched[cid] = {
                "center": center,
                "class": "unknown",
                "confidence": 0.0,
                "bbox": bbox,
                "reason": "likely_person_occlusion",
            }
            continue

        cls_idx, conf = classify_crop(model, tfm, crop, device)
        cls_name = classes[cls_idx]

        enriched[cid] = {
            "center": center,
            "class": cls_name,
            "confidence": conf,
            "bbox": bbox,
        }


    out_json = project_root / "output" / "hold_positions_enriched.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf8") as f:
        json.dump(enriched, f, indent=2)
    print(f"[enrich] Saved enriched holds JSON to {out_json}")

    overlay_out = project_root / "output" / "holds_enriched_overlay.jpg"
    draw_debug_overlay(img, holds, enriched, overlay_out)

    

if __name__ == "__main__":
    main()
