"""
Model predictor for serving.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.models.deeplob import create_model


class LOBPredictor:
    """Predictor for LOB mid-price movement."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        n_features: int = 144,
        n_classes: int = 3,
        window_size: int = 100,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to model checkpoint
            device: Device to run on
            n_features: Number of features
            n_classes: Number of classes
            window_size: Sequence window size
        """
        self.device = device
        self.n_features = n_features
        self.n_classes = n_classes
        self.window_size = window_size

        # Load model
        self.model = create_model(
            model_type="deeplob",
            n_features=n_features,
            n_classes=n_classes,
        )

        self._load_checkpoint(model_path)
        self.model.to(device)
        self.model.eval()

        self.class_names = ["down", "stationary", "up"]

    def _load_checkpoint(self, model_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"✓ Loaded model from {model_path}")

    def predict(
        self, features: np.ndarray, return_probabilities: bool = False
    ) -> Dict:
        """
        Make prediction.

        Args:
            features: Input features of shape (window_size, n_features)
            return_probabilities: Whether to return class probabilities

        Returns:
            Dictionary with prediction results
        """
        # Validate input
        if features.shape != (self.window_size, self.n_features):
            raise ValueError(
                f"Expected shape ({self.window_size}, {self.n_features}), "
                f"got {features.shape}"
            )

        # Prepare input
        x = features.T[np.newaxis, np.newaxis, :, :]  # (1, 1, n_features, window_size)
        x = torch.from_numpy(x).float().to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()

        result = {
            "predicted_class": int(predicted_class),
            "predicted_label": self.class_names[predicted_class],
            "confidence": float(probabilities[predicted_class].item()),
        }

        if return_probabilities:
            result["probabilities"] = {
                self.class_names[i]: float(probabilities[i].item())
                for i in range(self.n_classes)
            }

        return result

    def predict_batch(
        self, features_batch: np.ndarray, return_probabilities: bool = False
    ) -> List[Dict]:
        """
        Make batch predictions.

        Args:
            features_batch: Batch of features (batch_size, window_size, n_features)
            return_probabilities: Whether to return class probabilities

        Returns:
            List of prediction dictionaries
        """
        # Prepare input
        x = features_batch.transpose(0, 2, 1)[:, np.newaxis, :, :]
        x = torch.from_numpy(x).float().to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

        # Format results
        results = []
        for i in range(len(features_batch)):
            result = {
                "predicted_class": int(predicted_classes[i].item()),
                "predicted_label": self.class_names[predicted_classes[i].item()],
                "confidence": float(probabilities[i, predicted_classes[i]].item()),
            }

            if return_probabilities:
                result["probabilities"] = {
                    self.class_names[j]: float(probabilities[i, j].item())
                    for j in range(self.n_classes)
                }

            results.append(result)

        return results
