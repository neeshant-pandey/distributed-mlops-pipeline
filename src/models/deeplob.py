"""
DeepLOB: Deep Convolutional Neural Networks for Limit Order Books

Implementation based on:
Zhang, Z., Zohren, S., & Roberts, S. (2019).
DeepLOB: Deep convolutional neural networks for limit order books.
IEEE Transactions on Signal Processing, 67(11), 3001-3012.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLOB(nn.Module):
    """
    DeepLOB model for LOB mid-price movement prediction.

    Architecture:
    1. Convolutional layers for spatial feature extraction
    2. Inception module for multi-scale features
    3. LSTM for temporal modeling
    4. Fully connected layers for classification
    """

    def __init__(
        self,
        n_features: int = 144,
        n_classes: int = 3,
        conv_filters: int = 32,
        inception_filters: int = 32,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        dropout: float = 0.2,
    ):
        """
        Initialize DeepLOB model.

        Args:
            n_features: Number of input features (height dimension)
            n_classes: Number of output classes
            conv_filters: Number of filters in conv layers
            inception_filters: Number of filters in inception module
            lstm_hidden_size: Hidden size of LSTM
            lstm_num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(DeepLOB, self).__init__()

        self.n_features = n_features
        self.n_classes = n_classes

        # Convolutional layers
        # Conv1: (1, n_features, window_size) -> (32, n_features, window_size/2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_filters,
                kernel_size=(1, 2),
                stride=(1, 2),
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
        )

        # Conv2: (32, n_features, window_size/2) -> (32, n_features-3, window_size/2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=conv_filters,
                kernel_size=(4, 1),
                stride=(1, 1),
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
        )

        # Conv3: (32, n_features-7, window_size/2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=conv_filters,
                kernel_size=(4, 1),
                stride=(1, 1),
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_filters),
        )

        # Inception Module
        # Multiple parallel convolutions with different kernel sizes
        self.inception_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=inception_filters,
                kernel_size=(1, 1),
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(inception_filters),
        )

        self.inception_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=inception_filters,
                kernel_size=(3, 1),
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(inception_filters),
        )

        self.inception_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=inception_filters,
                kernel_size=(5, 1),
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(inception_filters),
        )

        # After inception: concatenate -> (inception_filters * 3, height, width)

        # LSTM
        # Input will be (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=inception_filters * 3,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, n_features, window_size)

        Returns:
            Output logits of shape (batch, n_classes)
        """
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Inception module
        x1 = self.inception_1(x)
        x2 = self.inception_2(x)
        x3 = self.inception_3(x)
        x = torch.cat([x1, x2, x3], dim=1)  # Concatenate along channel dimension

        # Prepare for LSTM
        # Current shape: (batch, channels, height, width)
        # LSTM expects: (batch, seq_len, features)
        # Use width as sequence length, flatten height and channels as features
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.reshape(batch_size, width, channels * height)

        # LSTM
        x, _ = self.lstm(x)
        # Take last output
        x = x[:, -1, :]

        # Fully connected
        x = self.fc(x)

        return x

    def get_num_params(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepLOBLite(nn.Module):
    """
    Lightweight version of DeepLOB for faster training and inference.
    """

    def __init__(
        self,
        n_features: int = 144,
        n_classes: int = 3,
        conv_filters: int = 16,
        lstm_hidden_size: int = 32,
        dropout: float = 0.2,
    ):
        """Initialize DeepLOB Lite model."""
        super(DeepLOBLite, self).__init__()

        self.n_features = n_features
        self.n_classes = n_classes

        # Simplified convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_filters, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters * 2),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_filters * 2,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, n_classes),
        )

    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)

        # Prepare for LSTM
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, width, channels * height)

        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        # FC
        x = self.fc(x)

        return x

    def get_num_params(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_type: str = "deeplob", **kwargs):
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('deeplob' or 'deeplob_lite')
        **kwargs: Model-specific arguments

    Returns:
        Model instance
    """
    if model_type == "deeplob":
        return DeepLOB(**kwargs)
    elif model_type == "deeplob_lite":
        return DeepLOBLite(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model
    print("Testing DeepLOB model...")

    batch_size = 4
    n_features = 144
    window_size = 100

    model = DeepLOB(n_features=n_features, n_classes=3)
    print(f"Total parameters: {model.get_num_params():,}")

    # Test forward pass
    x = torch.randn(batch_size, 1, n_features, window_size)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("✓ Model test passed!")

    # Test lite model
    print("\nTesting DeepLOB Lite model...")
    lite_model = DeepLOBLite(n_features=n_features, n_classes=3)
    print(f"Total parameters: {lite_model.get_num_params():,}")

    y_lite = lite_model(x)
    print(f"Output shape: {y_lite.shape}")
    print("✓ Lite model test passed!")
