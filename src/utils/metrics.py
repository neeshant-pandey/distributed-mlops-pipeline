"""
Custom metrics for model evaluation.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Tuple


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }

    # Add per-class metrics
    class_names = ["down", "stationary", "up"]
    for i, class_name in enumerate(class_names):
        if i < len(precision):
            metrics[f"precision_{class_name}"] = float(precision[i])
            metrics[f"recall_{class_name}"] = float(recall[i])
            metrics[f"f1_{class_name}"] = float(f1[i])

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Get detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Classification report string
    """
    class_names = ["down", "stationary", "up"]
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


class MetricsTracker:
    """Track metrics during training."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor, loss: float = None
    ):
        """
        Update metrics with batch results.

        Args:
            preds: Predicted labels (logits or class indices)
            targets: True labels
            loss: Batch loss value
        """
        # Convert to numpy
        if preds.dim() > 1:  # Logits
            preds = torch.argmax(preds, dim=1)

        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary of metrics
        """
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)

        metrics = compute_metrics(y_true, y_pred)

        if self.losses:
            metrics["loss"] = float(np.mean(self.losses))

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        return compute_confusion_matrix(y_true, y_pred)


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute directional accuracy (correct prediction of up/down, ignoring magnitude).

    Args:
        y_true: True labels (0=down, 1=stationary, 2=up)
        y_pred: Predicted labels

    Returns:
        Directional accuracy
    """
    # Convert to directional: -1=down, 0=stationary, 1=up
    y_true_dir = y_true - 1
    y_pred_dir = y_pred - 1

    # Count correct directions (ignoring stationary)
    mask = y_true_dir != 0
    if mask.sum() == 0:
        return 0.0

    correct = (np.sign(y_true_dir[mask]) == np.sign(y_pred_dir[mask])).sum()
    total = mask.sum()

    return float(correct / total)
