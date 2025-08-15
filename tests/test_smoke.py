"""Basic smoke tests for the PyTorch playground."""

from __future__ import annotations

import torch

from pytorch_playground.data import get_dataloaders
from pytorch_playground.model import SimpleNet


def test_get_dataloaders_fake():
    """Dataloaders should yield tensors with expected shapes."""
    train_loader, test_loader = get_dataloaders(batch_size=16, use_fake_data=True)
    images, labels = next(iter(train_loader))
    assert images.shape == (16, 1, 28, 28)
    assert labels.shape == (16,)
    images, labels = next(iter(test_loader))
    assert images.shape == (16, 1, 28, 28)
    assert labels.shape == (16,)


def test_model_forward():
    """Model forward pass returns logits for each class."""
    model = SimpleNet()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)
