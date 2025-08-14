"""Training utilities for the playground."""

from __future__ import annotations

from torch import nn
from torch.optim import SGD

from .data import get_dataloaders
from .model import SimpleNet


def train(num_epochs: int = 5, batch_size: int = 64, lr: float = 0.01) -> SimpleNet:
    """Train ``SimpleNet`` on the MNIST dataset."""
    train_loader, _ = get_dataloaders(batch_size=batch_size)
    model = SimpleNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            preds = model(images)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")
    return model


if __name__ == "__main__":
    train()
