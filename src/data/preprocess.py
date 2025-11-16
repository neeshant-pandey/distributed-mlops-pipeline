"""
Data preprocessing utilities for FI-2010 dataset.
"""

import numpy as np
from typing import Tuple, Optional


def normalize_features(
    X_train: np.ndarray, X_test: np.ndarray, method: str = "zscore"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize features using training statistics.

    Args:
        X_train: Training features
        X_test: Test features
        method: Normalization method ('zscore' or 'minmax')

    Returns:
        Tuple of (normalized_train, normalized_test)
    """
    if method == "zscore":
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero

        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

    elif method == "minmax":
        min_val = X_train.min(axis=0)
        max_val = X_train.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Avoid division by zero

        X_train_norm = (X_train - min_val) / range_val
        X_test_norm = (X_test - min_val) / range_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_train_norm, X_test_norm


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Compute class weights for handling imbalance.

    Args:
        labels: Array of class labels

    Returns:
        Array of weights for each class
    """
    unique, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum()  # Normalize to sum to 1
    return weights


def create_sequences(
    features: np.ndarray, labels: np.ndarray, window_size: int = 100, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences using sliding window.

    Args:
        features: Input features of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,)
        window_size: Size of each sequence
        stride: Step size for sliding window

    Returns:
        Tuple of (sequences, sequence_labels)
        - sequences: shape (n_sequences, window_size, n_features)
        - sequence_labels: shape (n_sequences,)
    """
    n_samples = len(features)
    sequences = []
    sequence_labels = []

    for i in range(window_size - 1, n_samples, stride):
        start_idx = i - window_size + 1
        end_idx = i + 1

        seq = features[start_idx:end_idx]
        label = labels[i]

        sequences.append(seq)
        sequence_labels.append(label)

    return np.array(sequences), np.array(sequence_labels)


def split_by_date(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data by chronological order (no shuffling for time series).

    Args:
        features: Input features
        labels: Labels
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n_samples = len(features)

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    X_train = features[:train_end]
    y_train = labels[:train_end]

    X_val = features[train_end:val_end]
    y_val = labels[train_end:val_end]

    X_test = features[val_end:]
    y_test = labels[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_data_statistics(features: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute dataset statistics.

    Args:
        features: Feature array
        labels: Label array

    Returns:
        Dictionary of statistics
    """
    stats = {
        "n_samples": len(features),
        "n_features": features.shape[1],
        "feature_mean": float(features.mean()),
        "feature_std": float(features.std()),
        "feature_min": float(features.min()),
        "feature_max": float(features.max()),
        "n_classes": len(np.unique(labels)),
        "class_distribution": {},
    }

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    for cls, count in zip(unique, counts):
        stats["class_distribution"][int(cls)] = {
            "count": int(count),
            "percentage": float(count / len(labels) * 100),
        }

    return stats
