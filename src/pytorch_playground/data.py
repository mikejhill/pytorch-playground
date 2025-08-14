"""Dataset utilities for the playground."""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    batch_size: int = 64, use_fake_data: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Return training and test dataloaders.

    Args:
    ----
        batch_size: Number of items per batch.
        use_fake_data: If True, return loaders built from ``datasets.FakeData`` to
            avoid downloading the real dataset. Helpful for tests.
    """
    transform = transforms.ToTensor()
    if use_fake_data:
        train_dataset = datasets.FakeData(
            size=64,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        test_dataset = datasets.FakeData(
            size=64,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
    else:
        train_dataset = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
