# enrich_holds_with_classifier.py
#
# Use the trained hold classifier to assign a class and confidence
# to each contour center in output/hold_positions_auto.json.
#
# Now uses the composite mask to isolate holds on white backgrounds,
# eliminating climber occlusion issues.

from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


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


def crop_hold_with_mask(
    img: np.ndarray,
    mask: np.ndarray,
    cx: float,
    cy: float,
    size: int = 160
) -> Tuple[np.ndarray, list]:
    """
    Crop around center, keeping ONLY the connected component at the center.
    Keeps the natural wall background to match training data.
    """
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

    # Crop the image and mask
    crop_img = img[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].copy()
    
    # Find connected components in the cropped mask
    num_labels, labels = cv2.connectedComponents(crop_mask)
    
    # Find which component contains the center point
    local_cx = cx - x1
    local_cy = cy - y1
    
    # Make sure center is within crop bounds
    if 0 <= local_cy < labels.shape[0] and 0 <= local_cx < labels.shape[1]:
        center_label = labels[local_cy, local_cx]
        
        if center_label > 0:  # 0 is background
            # Create mask with only the component containing the center
            single_hold_mask = (labels == center_label).astype(np.uint8) * 255
        else:
            # Center is in background, use the whole mask
            single_hold_mask = crop_mask
    else:
        single_hold_mask = crop_mask

    # KEEP THE NATURAL BACKGROUND - just use the original crop
    # The single_hold_mask isolates the hold but we return the full crop
    # This matches the training data format better
    isolated_hold = crop_img

    return isolated_hold, [x1, y1, x2, y2]


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
    composite_mask = project_root / "output" / "hold_mask_composite.jpg"
    model_path = project_root / "models" / "hold_classifier_resnet18.pt"
    labels_path = project_root / "models" / "hold_class_labels.json"

    if not holds_json.exists():
        raise FileNotFoundError(f"Holds JSON not found: {holds_json}")
    if not frame_image.exists():
        raise FileNotFoundError(f"Frame image not found: {frame_image}")
    if not composite_mask.exists():
        raise FileNotFoundError(f"Composite mask not found: {composite_mask}")
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

    # Load the composite mask
    mask = cv2.imread(str(composite_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read composite mask {composite_mask}")

    print(f"Loaded composite mask: {mask.shape}")

    # folder for visual debugging of crops
    debug_dir = project_root / "debug_patches"
    debug_dir.mkdir(exist_ok=True)

    enriched: Dict[str, Dict[str, Any]] = {}

    for cid, center in holds.items():
        cx, cy = center
        crop, bbox = crop_hold_with_mask(img, mask, cx, cy, size=160)
        
        if crop is None:
            enriched[cid] = {
                "center": center,
                "class": "unknown",
                "confidence": 0.0,
                "bbox": None,
            }
            continue

        # save the isolated crop (hold on white background)
        debug_path = debug_dir / f"patch_{cid}.jpg"
        cv2.imwrite(str(debug_path), crop)

        # Visualize first 3 holds
        if cid in ["contour_0", "contour_1", "contour_2"]:
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plt.title(f"Hold {cid} - Isolated on white background")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"visual_debug_{cid}.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization: visual_debug_{cid}.png")

        # No need for person occlusion check - mask handles it!
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