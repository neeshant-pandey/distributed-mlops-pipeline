"""
Tests for dataset classes.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch


class TestLOBDataset:
    """Test LOB Dataset."""

    @patch('src.data.dataset.FI2010Downloader')
    def test_dataset_creation(self, mock_downloader):
        """Test dataset can be created."""
        from src.data.dataset import LOBDataset

        # Mock data
        mock_features = np.random.randn(1000, 144).astype(np.float32)
        mock_labels = np.random.randint(-1, 2, size=(1000, 5)).astype(np.int64)

        mock_downloader.return_value.load_data.return_value = (
            mock_features,
            mock_labels,
        )

        dataset = LOBDataset(
            data_path="dummy",
            split="train",
            horizon_index=0,
            window_size=100,
        )

        assert len(dataset) > 0

    @patch('src.data.dataset.FI2010Downloader')
    def test_getitem(self, mock_downloader):
        """Test getting a single item."""
        from src.data.dataset import LOBDataset

        # Mock data
        n_samples = 1000
        mock_features = np.random.randn(n_samples, 144).astype(np.float32)
        mock_labels = np.random.randint(-1, 2, size=(n_samples, 5)).astype(
            np.int64
        )

        mock_downloader.return_value.load_data.return_value = (
            mock_features,
            mock_labels,
        )

        dataset = LOBDataset(
            data_path="dummy",
            split="train",
            horizon_index=0,
            window_size=100,
        )

        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (1, 144, 100)
        assert y.dim() == 0  # Scalar


class TestDataPreprocessing:
    """Test data preprocessing functions."""

    def test_normalize_features(self):
        """Test feature normalization."""
        from src.data.preprocess import normalize_features

        X_train = np.random.randn(100, 10).astype(np.float32)
        X_test = np.random.randn(50, 10).astype(np.float32)

        X_train_norm, X_test_norm = normalize_features(X_train, X_test)

        # Check mean and std
        assert np.abs(X_train_norm.mean()) < 1e-5
        assert np.abs(X_train_norm.std() - 1.0) < 1e-5

    def test_compute_class_weights(self):
        """Test class weight computation."""
        from src.data.preprocess import compute_class_weights

        labels = np.array([0, 0, 0, 1, 1, 2])
        weights = compute_class_weights(labels)

        assert len(weights) == 3
        assert np.isclose(weights.sum(), 1.0)
