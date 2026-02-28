# enrich_holds_with_classifier_multiframe.py
#
# Extract hold crops from multiple video frames and pick the clearest one
# (the one with the most visible hold pixels, least occlusion)

from pathlib import Path
import json
from typing import Dict, Any, List, Tuple

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

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, classes, tfm


def load_video_frames(video_path: str, num_frames: int = 5) -> List[np.ndarray]:
    """Load evenly spaced frames from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")

    frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    print(f"[enrich] Loaded {len(frames)} frames from video")
    return frames


def get_best_crop_for_hold(
    frames: List[np.ndarray],
    mask: np.ndarray,
    cx: float,
    cy: float,
    base_size: int = 160,
    pad: int = 8,
) -> Tuple[np.ndarray, list]:
    """
    Try cropping the hold from each frame, pick the one with most visible hold pixels.
    """
    cx = int(round(cx))
    cy = int(round(cy))
    half = base_size // 2

    best_crop = None
    best_bbox = None
    best_score = -1.0

    for frame in frames:
        h, w = frame.shape[:2]

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        if x2 <= x1 or y2 <= y1:
            continue

        crop_img = frame[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2].copy()

        num_labels, labels = cv2.connectedComponents(crop_mask)

        local_cx = cx - x1
        local_cy = cy - y1

        if 0 <= local_cy < labels.shape[0] and 0 <= local_cx < labels.shape[1]:
            center_label = labels[local_cy, local_cx]
            if center_label > 0:
                single_hold_mask = (labels == center_label).astype(np.uint8) * 255
            else:
                single_hold_mask = crop_mask
        else:
            single_hold_mask = crop_mask

        hold_pixels = np.count_nonzero(single_hold_mask)
        if hold_pixels == 0:
            continue

        ys, xs = np.where(single_hold_mask > 0)
        min_x = xs.min()
        max_x = xs.max()
        min_y = ys.min()
        max_y = ys.max()

        tight_x1 = max(0, min_x - pad)
        tight_y1 = max(0, min_y - pad)
        tight_x2 = min(crop_img.shape[1] - 1, max_x + pad)
        tight_y2 = min(crop_img.shape[0] - 1, max_y + pad)

        min_side = 32
        if tight_x2 - tight_x1 + 1 < min_side:
            extra = (min_side - (tight_x2 - tight_x1 + 1)) // 2
            tight_x1 = max(0, tight_x1 - extra)
            tight_x2 = min(crop_img.shape[1] - 1, tight_x2 + extra)
        if tight_y2 - tight_y1 + 1 < min_side:
            extra = (min_side - (tight_y2 - tight_y1 + 1)) // 2
            tight_y1 = max(0, tight_y1 - extra)
            tight_y2 = min(crop_img.shape[0] - 1, tight_y2 + extra)

        tight_crop_img = crop_img[tight_y1:tight_y2 + 1, tight_x1:tight_x2 + 1]
        tight_mask = single_hold_mask[tight_y1:tight_y2 + 1, tight_x1:tight_x2 + 1]

        ycrcb = cv2.cvtColor(tight_crop_img, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower, upper)

        skin_in_hold = cv2.bitwise_and(skin_mask, skin_mask, mask=tight_mask)
        visible_hold_mask = tight_mask.copy()
        visible_hold_mask[skin_in_hold > 0] = 0

        visible_pixels = np.count_nonzero(visible_hold_mask)

        if visible_pixels < 0.2 * hold_pixels:
            continue

        gray = cv2.cvtColor(tight_crop_img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray[visible_hold_mask > 0]) if visible_pixels > 0 else 0.0

        score = visible_pixels * (mean_brightness / 255.0)

        if score > best_score:
            best_score = score
            best_crop = tight_crop_img

            full_x1 = x1 + tight_x1
            full_y1 = y1 + tight_y1
            full_x2 = x1 + tight_x2 + 1
            full_y2 = y1 + tight_y2 + 1

            best_bbox = [int(full_x1), int(full_y1), int(full_x2), int(full_y2)]

    return best_crop, best_bbox


def classify_crop(model, tfm, crop: np.ndarray, device: torch.device):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = tfm(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)

    return int(idx.item()), float(conf.item())


def main(video_path: str, output_dir: str = "output"):
    """
    Main hold classification function.
    
    Args:
        video_path: Path to video file
        output_dir: Directory containing hold detection outputs
    """
    project_root = Path(".")
    output_path = Path(output_dir)

    # Look for holds JSON - try split version first
    holds_json = output_path / "hold_positions_auto_split.json"
    if not holds_json.exists():
        holds_json = output_path / "hold_positions_auto.json"
    
    composite_mask = output_path / "hold_mask_composite.jpg"
    model_path = project_root / "models" / "hold_classifier_resnet18.pt"
    labels_path = project_root / "models" / "hold_class_labels.json"

    if not holds_json.exists():
        print(f"[enrich] Holds JSON not found: {holds_json}")
        return
    if not Path(video_path).exists():
        print(f"[enrich] Video not found: {video_path}")
        return
    if not composite_mask.exists():
        print(f"[enrich] Composite mask not found: {composite_mask}")
        return
    if not model_path.exists():
        print(f"[enrich] Model not found: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[enrich] Using device: {device}")

    holds = load_holds(holds_json)
    model, classes, tfm = build_model(model_path, labels_path, device)
    print(f"[enrich] Classifier classes: {classes}")

    frames = load_video_frames(video_path, num_frames=5)
    
    mask = cv2.imread(str(composite_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[enrich] Could not read composite mask")
        return

    debug_dir = output_path / "debug_patches"
    debug_dir.mkdir(exist_ok=True)

    enriched: Dict[str, Dict[str, Any]] = {}

    for cid, center in holds.items():
        cx, cy = center
        
        crop, bbox = get_best_crop_for_hold(frames, mask, cx, cy, base_size=160)
        
        if crop is None:
            enriched[cid] = {
                "center": center,
                "class": "unknown",
                "confidence": 0.0,
                "bbox": None,
            }
            continue

        debug_path = debug_dir / f"patch_{cid}.jpg"
        cv2.imwrite(str(debug_path), crop)

        cls_idx, conf = classify_crop(model, tfm, crop, device)
        cls_name = classes[cls_idx]

        enriched[cid] = {
            "center": center,
            "class": cls_name,
            "confidence": conf,
            "bbox": bbox,
        }
        
        print(f"[enrich] {cid}: {cls_name} ({conf:.3f})")

    out_json = output_path / "hold_positions_enriched.json"
    with out_json.open("w", encoding="utf8") as f:
        json.dump(enriched, f, indent=2)
    print(f"[enrich] ✅ Saved to {out_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    main(args.video_path, args.output)