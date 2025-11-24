# move_classifier/train_classifier.py

from pathlib import Path
import numpy as np

from move_classifier.move_labels import CLASS_NAMES, INDEX_TO_LABEL

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DATASET_PATH = Path("move_classifier/move_pose_dataset.npz")
MODEL_DIR = Path("move_classifier/saved_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "pose_move_classifier.keras"


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    data = np.load(DATASET_PATH, allow_pickle=True)
    X = data["X"]  # shape: (N, 99)
    y = data["y"]  # shape: (N,)
    paths = data["paths"]

    print("Loaded dataset:")
    print("  X shape:", X.shape)
    print("  y shape:", y.shape)

    # Show class counts
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution (index: count, label):")
    for idx, count in zip(unique, counts):
        label = INDEX_TO_LABEL.get(int(idx), f"unknown_{idx}")
        print(f"  {idx}: {count}  ({label})")

    return X, y


def build_model(input_dim: int, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    X, y = load_dataset()

    num_samples, input_dim = X.shape
    num_classes = len(CLASS_NAMES)

    # Check how many distinct labels we actually have
    present_labels = np.unique(y)
    if len(present_labels) < 2:
        print("\nWARNING: Only one class present in the dataset.")
        print("Training will not be meaningful until you add images for other move types")
        print("and rerun dataset_builder.py.")
        # You can comment this return out if you still want to test-run training.
        return

    print(f"\nInput dim: {input_dim}")
    print(f"Num classes (configured): {num_classes}")
    print("Labels present in this dataset (indices):", present_labels)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/val shapes:")
    print("  X_train:", X_train.shape, " y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape, " y_val:  ", y_val.shape)

    # Class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(np.unique(y_train), class_weights)}
    print("\nClass weights:", class_weight_dict)

    model = build_model(input_dim=input_dim, num_classes=num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nTraining complete. Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
