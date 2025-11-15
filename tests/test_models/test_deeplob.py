"""
Tests for DeepLOB model.
"""

import pytest
import torch
from src.models.deeplob import DeepLOB, DeepLOBLite, create_model


class TestDeepLOB:
    """Test DeepLOB model."""

    def test_model_creation(self):
        """Test model can be created."""
        model = DeepLOB(n_features=144, n_classes=3)
        assert model is not None
        assert model.n_features == 144
        assert model.n_classes == 3

    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = DeepLOB(n_features=144, n_classes=3)
        batch_size = 4
        window_size = 100

        x = torch.randn(batch_size, 1, 144, window_size)
        output = model(x)

        assert output.shape == (batch_size, 3)

    def test_output_range(self):
        """Test output logits are valid."""
        model = DeepLOB(n_features=144, n_classes=3)
        x = torch.randn(2, 1, 144, 100)
        output = model(x)

        # Check no NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_num_parameters(self):
        """Test parameter count."""
        model = DeepLOB(n_features=144, n_classes=3)
        num_params = model.get_num_params()

        assert num_params > 0
        assert isinstance(num_params, int)


class TestDeepLOBLite:
    """Test DeepLOB Lite model."""

    def test_lite_model_creation(self):
        """Test lite model can be created."""
        model = DeepLOBLite(n_features=144, n_classes=3)
        assert model is not None

    def test_lite_forward_pass(self):
        """Test lite model forward pass."""
        model = DeepLOBLite(n_features=144, n_classes=3)
        x = torch.randn(2, 1, 144, 100)
        output = model(x)

        assert output.shape == (2, 3)

    def test_lite_smaller_than_full(self):
        """Test lite model has fewer parameters."""
        model_full = DeepLOB(n_features=144, n_classes=3)
        model_lite = DeepLOBLite(n_features=144, n_classes=3)

        assert model_lite.get_num_params() < model_full.get_num_params()


class TestModelFactory:
    """Test model factory function."""

    def test_create_deeplob(self):
        """Test creating DeepLOB via factory."""
        model = create_model("deeplob", n_features=144, n_classes=3)
        assert isinstance(model, DeepLOB)

    def test_create_lite(self):
        """Test creating DeepLOB Lite via factory."""
        model = create_model("deeplob_lite", n_features=144, n_classes=3)
        assert isinstance(model, DeepLOBLite)

    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError):
            create_model("invalid_model")
