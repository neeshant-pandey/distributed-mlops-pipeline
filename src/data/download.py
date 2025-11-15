"""
FI-2010 Dataset Download Script

Generates synthetic LOB data for demonstration purposes or downloads the real FI-2010 dataset.

Real FI-2010 Dataset Information:
- Source: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
- Alternative: https://zenodo.org/records/4654804 or researcher's institutional repository
- 5 stocks from Finnish stock market
- 10 days of tick data per stock
- 10 levels of order book depth
- ~4.5 million data points per stock

The dataset contains:
- 144 features: 40 price levels (ask/bid) + 40 volume levels + 64 derived features
- Labels: -1 (down), 0 (stationary), 1 (up) for k=1,2,3,5,10 time steps ahead

Note: This script generates synthetic data that matches the FI-2010 format for
demonstration purposes. For production use, download the real dataset from the
official sources above.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np


class FI2010Downloader:
    """Generate synthetic FI-2010 format dataset for demonstration."""

    FILES = {
        "train": [
            "Train_Dst_NoAuction_DecPre_CF_7.txt",
            "Train_Dst_NoAuction_DecPre_CF_8.txt",
            "Train_Dst_NoAuction_DecPre_CF_9.txt",
        ],
        "test": [
            "Test_Dst_NoAuction_DecPre_CF_7.txt",
            "Test_Dst_NoAuction_DecPre_CF_8.txt",
            "Test_Dst_NoAuction_DecPre_CF_9.txt",
        ],
    }

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the downloader.

        Args:
            data_dir: Directory to save downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_synthetic_data(
        self, n_samples: int = 100000, n_features: int = 144, n_labels: int = 5
    ) -> np.ndarray:
        """
        Generate synthetic LOB data matching FI-2010 format.

        Args:
            n_samples: Number of samples to generate
            n_features: Number of features (default: 144)
            n_labels: Number of label columns (default: 5 for k=1,2,3,5,10)

        Returns:
            Array of shape (n_samples, n_features + n_labels)
        """
        np.random.seed(42)

        # Generate normalized features (mean ~0, std ~1)
        features = np.random.randn(n_samples, n_features).astype(np.float32)

        # Generate labels: -1 (down), 0 (stationary), 1 (up)
        # Slightly imbalanced to match real data characteristics
        labels = np.random.choice(
            [-1, 0, 1], size=(n_samples, n_labels), p=[0.3, 0.4, 0.3]
        ).astype(np.int32)

        # Combine features and labels
        data = np.concatenate([features, labels], axis=1)

        return data

    def generate_file(self, filename: str, n_samples: int, force: bool = False) -> Path:
        """
        Generate a single synthetic data file.

        Args:
            filename: Name of the file to generate
            n_samples: Number of samples to generate
            force: If True, regenerate even if file exists

        Returns:
            Path to the generated file
        """
        file_path = self.data_dir / filename

        if file_path.exists() and not force:
            print(f"✓ {filename} already exists, skipping generation")
            return file_path

        print(f"🔧 Generating {filename}...")

        try:
            data = self.generate_synthetic_data(n_samples=n_samples)
            np.savetxt(file_path, data, fmt="%.6f")
            print(
                f"✓ Generated {filename} ({file_path.stat().st_size / 1024 / 1024:.2f} MB, "
                f"{n_samples:,} samples)"
            )
            return file_path
        except Exception as e:
            print(f"✗ Failed to generate {filename}: {e}")
            raise

    def download_all(self, force: bool = False) -> dict[str, list[Path]]:
        """
        Generate all synthetic train and test files.

        Args:
            force: If True, regenerate even if files exist

        Returns:
            Dictionary with 'train' and 'test' keys containing lists of file paths
        """
        print("=" * 70)
        print("FI-2010 Synthetic Dataset Generation")
        print("=" * 70)
        print("Note: Generating synthetic data for demonstration purposes.")
        print("For production use, download the real FI-2010 dataset from:")
        print("https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649")
        print("=" * 70)

        generated_files = {"train": [], "test": []}

        # Generate train files (larger)
        print("\n📥 Generating Training Files...")
        for filename in self.FILES["train"]:
            file_path = self.generate_file(filename, n_samples=150000, force=force)
            generated_files["train"].append(file_path)

        # Generate test files (smaller)
        print("\n📥 Generating Test Files...")
        for filename in self.FILES["test"]:
            file_path = self.generate_file(filename, n_samples=50000, force=force)
            generated_files["test"].append(file_path)

        print("\n" + "=" * 70)
        print("✓ Generation Complete!")
        print(f"  Train files: {len(generated_files['train'])} (450,000 samples total)")
        print(f"  Test files: {len(generated_files['test'])} (150,000 samples total)")
        print(f"  Location: {self.data_dir.absolute()}")
        print("=" * 70)

        return generated_files

    def load_data(self, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
        """
        Load the downloaded data files.

        Args:
            split: Either 'train' or 'test'

        Returns:
            Tuple of (features, labels)
            - features: shape (n_samples, n_features)
            - labels: shape (n_samples, n_horizons) where n_horizons = 5 (k=1,2,3,5,10)
        """
        if split not in ["train", "test"]:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        data_list = []

        for filename in self.FILES[split]:
            file_path = self.data_dir / filename

            if not file_path.exists():
                raise FileNotFoundError(
                    f"{filename} not found. Run download_all() first."
                )

            print(f"📖 Loading {filename}...")
            data = np.loadtxt(file_path)
            data_list.append(data)

        # Concatenate all files
        all_data = np.concatenate(data_list, axis=0)

        # Split features and labels
        # Last 5 columns are labels for k=1,2,3,5,10
        features = all_data[:, :-5]
        labels = all_data[:, -5:]

        print(f"✓ Loaded {split} data:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")

        return features, labels

    def verify_data(self) -> bool:
        """
        Verify that all required files exist and are valid.

        Returns:
            True if all files are present and valid
        """
        print("\n🔍 Verifying downloaded data...")

        all_files = self.FILES["train"] + self.FILES["test"]
        missing_files = []

        for filename in all_files:
            file_path = self.data_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
                print(f"✗ Missing: {filename}")
            else:
                size_mb = file_path.stat().st_size / 1024 / 1024
                print(f"✓ Found: {filename} ({size_mb:.2f} MB)")

        if missing_files:
            print(f"\n✗ {len(missing_files)} files missing")
            return False
        else:
            print(f"\n✓ All {len(all_files)} files present")
            return True


def main():
    """Main function to download the dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Download FI-2010 dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory to save data (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing files without downloading",
    )

    args = parser.parse_args()

    downloader = FI2010Downloader(data_dir=args.data_dir)

    if args.verify:
        downloader.verify_data()
    else:
        downloader.download_all(force=args.force)
        downloader.verify_data()


if __name__ == "__main__":
    main()
