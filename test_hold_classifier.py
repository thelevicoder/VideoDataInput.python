# test_hold_classifier.py
#
# Use your trained ResNet18 hold classifier to predict the class
# for a single image of a hold.

from pathlib import Path
import argparse
import json

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


def load_model_and_labels(models_dir: Path, device):
    model_path = models_dir / "hold_classifier_resnet18.pt"
    labels_path = models_dir / "hold_class_labels.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    with labels_path.open("r", encoding="utf8") as f:
        label_data = json.load(f)
    classes = label_data["classes"]

    print("Loaded classes:", classes)

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


def predict_image(img_path: Path, model, classes, tfm, device):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image {img_path}")

    # convert BGR to RGB for PIL
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    x = tfm(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)

    cls_idx = int(idx.item())
    cls_name = classes[cls_idx]
    conf_val = float(conf.item())

    return cls_name, conf_val, probs.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Test hold classifier on a single image"
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Path to an image of a single hold",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    models_dir = Path("models")
    model, classes, tfm = load_model_and_labels(models_dir, device)

    print(f"Running prediction on {img_path}")
    cls_name, conf_val, probs = predict_image(
        img_path, model, classes, tfm, device
    )

    print()
    print("Prediction result")
    print("-----------------")
    print(f"Class:      {cls_name}")
    print(f"Confidence: {conf_val:.4f}")
    print()
    print("Raw probabilities per class:")
    for i, p in enumerate(probs):
        print(f"  {classes[i]}: {p:.4f}")


if __name__ == "__main__":
    main()
