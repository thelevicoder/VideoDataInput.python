from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from tensorflow import keras
from move_classifier.move_labels import INDEX_TO_LABEL

MODEL_PATH = Path(__file__).resolve().parent / "saved_models" / "pose_move_classifier.keras"


class PoseMoveClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model = keras.models.load_model(model_path)

    @staticmethod
    def features_from_landmarks(landmarks: Sequence) -> np.ndarray:
        """
        landmarks: iterable of MediaPipe pose landmarks
        Returns feature vector shape (99,) with [x, y, visibility] for 33 landmarks.
        """
        coords = np.array(
            [[lm.x, lm.y, lm.visibility] for lm in landmarks],
            dtype=np.float32,
        )  # shape (33, 3)
        return coords.flatten()

    def predict_from_landmarks(self, landmarks: Sequence) -> Tuple[str, np.ndarray]:
        """
        Returns (predicted_label, probabilities_array)
        """
        features = self.features_from_landmarks(landmarks)
        x = features.reshape(1, -1)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = INDEX_TO_LABEL[idx]
        return label, probs
