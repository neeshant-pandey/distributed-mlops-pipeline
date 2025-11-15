"""
PyTorch Dataset for FI-2010 Limit Order Book data.

Implements sliding window approach for temporal sequence modeling.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LOBDataset(Dataset):
    """PyTorch Dataset for Limit Order Book data with sliding window."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        horizon_index: int = 0,
        window_size: int = 100,
        normalize: bool = False,
        device: str = "cpu",
    ):
        """
        Initialize LOB Dataset.

        Args:
            data_path: Path to data directory
            split: Either 'train' or 'test'
            horizon_index: Which prediction horizon to use (0-4 for k=1,2,3,5,10)
            window_size: Number of timesteps in each sequence
            normalize: Whether to normalize features (data is already normalized)
            device: Device to load data on
        """
        self.data_path = Path(data_path)
        self.split = split
        self.horizon_index = horizon_index
        self.window_size = window_size
        self.normalize = normalize
        self.device = device

        # Load data
        self.features, self.labels = self._load_data()

        # Create sliding windows
        self.windows = self._create_windows()

        print(f"✓ Loaded {split} dataset:")
        print(f"  Total samples: {len(self.windows):,}")
        print(f"  Window size: {window_size}")
        print(f"  Prediction horizon: k={[1, 2, 3, 5, 10][horizon_index]}")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and concatenate all data files."""
        from src.data.download import FI2010Downloader

        downloader = FI2010Downloader(data_dir=str(self.data_path))
        features, labels = downloader.load_data(split=self.split)

        # Extract single horizon
        labels = labels[:, self.horizon_index]

        # Convert labels to 0, 1, 2 (from -1, 0, 1)
        labels = labels + 1

        return features.astype(np.float32), labels.astype(np.int64)

    def _create_windows(self) -> np.ndarray:
        """Create sliding window indices."""
        n_samples = len(self.features)
        valid_indices = np.arange(self.window_size - 1, n_samples)
        return valid_indices

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (features, label)
            - features: shape (1, window_size, n_features)
            - label: scalar class index
        """
        end_idx = self.windows[idx]
        start_idx = end_idx - self.window_size + 1

        # Get window of features
        x = self.features[start_idx : end_idx + 1]  # Shape: (window_size, n_features)

        # Get label at end of window
        y = self.labels[end_idx]

        # Reshape for CNN: (channels, height, width)
        # height = n_features, width = window_size, channels = 1
        x = x.T[np.newaxis, :, :]  # Shape: (1, n_features, window_size)

        # Convert to tensors
        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalance.

        Returns:
            Tensor of class weights
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum()  # Normalize
        return torch.tensor(weights, dtype=torch.float32)


class LOBDataModule:
    """Data module for LOB dataset with train/val/test splits."""

    def __init__(
        self,
        data_path: str = "data/raw",
        horizon_index: int = 0,
        window_size: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
    ):
        """
        Initialize data module.

        Args:
            data_path: Path to data directory
            horizon_index: Which prediction horizon to use (0-4)
            window_size: Number of timesteps in sequence
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            val_split: Fraction of train data to use for validation
        """
        self.data_path = data_path
        self.horizon_index = horizon_index
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        # Will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """Create train/val/test datasets."""
        # Load full training data
        full_train = LOBDataset(
            data_path=self.data_path,
            split="train",
            horizon_index=self.horizon_index,
            window_size=self.window_size,
        )

        # Split into train and validation
        n_val = int(len(full_train) * self.val_split)
        n_train = len(full_train) - n_val

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

        # Load test data
        self.test_dataset = LOBDataset(
            data_path=self.data_path,
            split="test",
            horizon_index=self.horizon_index,
            window_size=self.window_size,
        )

        print(f"\n✓ Data splits created:")
        print(f"  Train: {len(self.train_dataset):,} samples")
        print(f"  Val:   {len(self.val_dataset):,} samples")
        print(f"  Test:  {len(self.test_dataset):,} samples")

    def train_dataloader(self):
        """Create training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
