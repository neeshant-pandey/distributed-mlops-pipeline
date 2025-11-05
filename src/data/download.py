"""
FI-2010 Dataset Download Script

Downloads the FI-2010 (Finnish Intraday) dataset for Limit Order Book mid-price prediction.

Dataset Information:
- Source: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
- Alternative: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
- 5 stocks from Finnish stock market
- 10 days of tick data per stock
- 10 levels of order book depth
- ~4.5 million data points per stock

The dataset contains:
- BenchmarkDatasets/NoAuction/[1-5].[Stock1-Stock5]/
  - Each file contains normalized LOB data
  - 144 features: 40 price levels (ask/bid) + 40 volume levels + derived features
  - Labels: -1 (down), 0 (stationary), 1 (up) for k time steps ahead
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np


class FI2010Downloader:
    """Download and extract FI-2010 dataset."""

    # Direct download URLs for FI-2010 data files
    # These are the preprocessed data files from the DeepLOB paper
    BASE_URL = "https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data"

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

    def download_file(self, filename: str, force: bool = False) -> Path:
        """
        Download a single file from the dataset.

        Args:
            filename: Name of the file to download
            force: If True, re-download even if file exists

        Returns:
            Path to the downloaded file
        """
        file_path = self.data_dir / filename

        if file_path.exists() and not force:
            print(f"✓ {filename} already exists, skipping download")
            return file_path

        url = f"{self.BASE_URL}/{filename}"
        print(f"⬇ Downloading {filename}...")
        print(f"   URL: {url}")

        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"✓ Downloaded {filename} ({file_path.stat().st_size / 1024 / 1024:.2f} MB)")
            return file_path
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            raise

    def download_all(self, force: bool = False) -> dict[str, list[Path]]:
        """
        Download all train and test files.

        Args:
            force: If True, re-download even if files exist

        Returns:
            Dictionary with 'train' and 'test' keys containing lists of file paths
        """
        print("=" * 70)
        print("FI-2010 Dataset Download")
        print("=" * 70)

        downloaded_files = {"train": [], "test": []}

        # Download train files
        print("\n📥 Downloading Training Files...")
        for filename in self.FILES["train"]:
            file_path = self.download_file(filename, force=force)
            downloaded_files["train"].append(file_path)

        # Download test files
        print("\n📥 Downloading Test Files...")
        for filename in self.FILES["test"]:
            file_path = self.download_file(filename, force=force)
            downloaded_files["test"].append(file_path)

        print("\n" + "=" * 70)
        print("✓ Download Complete!")
        print(f"  Train files: {len(downloaded_files['train'])}")
        print(f"  Test files: {len(downloaded_files['test'])}")
        print(f"  Location: {self.data_dir.absolute()}")
        print("=" * 70)

        return downloaded_files

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
