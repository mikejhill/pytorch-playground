"""Model definitions for the playground."""

from __future__ import annotations

from torch import nn


class SimpleNet(nn.Module):
    """A small neural network for MNIST digits."""

    def __init__(self) -> None:
        """Create network layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):  # type: ignore[override]
        """Compute logits for a batch of images."""
        return self.net(x)
