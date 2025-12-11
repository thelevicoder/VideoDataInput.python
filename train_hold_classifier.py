# train_hold_classifier.py
#
# Train a simple ResNet18 classifier on the Kaggle hold dataset.
# Uses patches created by convert_hold_detection_to_patches.py:
# dataset/HoldClass/Final_Dataset/patches/train/<class_name>/*.jpg

from pathlib import Path
import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # use the cropped patches
    data_root = (
        Path("dataset") / "HoldClass" / "Final_Dataset" / "patches"
    )

    train_dir = data_root / "train"
    val_dir = data_root / "valid"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Valid dir not found: {val_dir}")

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=0
    )

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    # ResNet18 backbone
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 15
    best_val_acc = 0.0
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # show progress every 20 batches
            if (i + 1) % 20 == 0:
                print(
                    f"  [batch {i+1}/{len(train_loader)}] "
                    f"loss: {running_loss / (i+1):.4f}, "
                    f"acc: {100.0 * correct / total:.2f}%"
                )

        # end of epoch stats
        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)
        print(
            f"  -> Training loss {train_loss:.4f}, accuracy {train_acc:.2f}%"
        )

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        print(
            f"  -> Validation loss {val_loss:.4f}, accuracy {val_acc:.2f}%"
        )

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = save_dir / "hold_classifier_resnet18.pt"
            torch.save(model.state_dict(), model_path)
            print(
                f"  -> Saved new best model with val_acc {best_val_acc:.2f}% "
                f"to {model_path}"
            )

    # Save label mapping
        # Use your manual mapping for nicer names
    nice_names = [
        "crimp_or_foot",  # class_0
        "jug",            # class_1
        "pinch",          # class_2
        "pocket",         # class_3
        "sloper",         # class_4
        "volume",         # class_5
    ]

    assert len(nice_names) == len(train_ds.classes), "Class count mismatch"

    labels_path = save_dir / "hold_class_labels.json"
    with labels_path.open("w", encoding="utf8") as f:
        json.dump({"classes": nice_names}, f, indent=2)

    print("Done. Best model and labels saved in models/")



if __name__ == "__main__":
    main()
