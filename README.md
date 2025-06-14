# PyTorch Playground

This repository is a starting point for experimenting with PyTorch and learning how to build machine learning models.

## Project Idea

A great beginner project is to train a neural network that can recognize handwritten digits using the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. The goal is to classify input images of digits (0–9) into the correct category. MNIST is small enough to work on most computers and is commonly used as a first step into deep learning.

### Why MNIST?
- It is well-known and simple, so you can focus on PyTorch basics.
- Each image is only 28×28 pixels in grayscale, making training relatively quick.
- Tutorials and code examples are widely available if you get stuck.

## Getting Started

1. **Install Python 3.8 or newer.** Creating a virtual environment is recommended:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

2. **Install PyTorch and supporting libraries.** Visit <https://pytorch.org> for the command specific to your system. A typical CPU-only install looks like:

   ```bash
   pip install torch torchvision
   ```

3. **Clone this repository** (if you haven't already) and add your own training script. You can start from the following minimal example that trains a simple network:

   ```python
   import torch
   from torch import nn
   from torchvision import datasets, transforms

   train_data = datasets.MNIST(
       root="./data", train=True, download=True,
       transform=transforms.ToTensor()
   )
   train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

   model = nn.Sequential(
       nn.Flatten(),
       nn.Linear(28 * 28, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   )
   loss_fn = nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

   for epoch in range(5):
       for images, labels in train_loader:
           preds = model(images)
           loss = loss_fn(preds, labels)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

       print(f"Epoch {epoch+1}, loss={loss.item():.4f}")
   ```

4. **Run your script** to start training and watch the loss decrease as the model learns. From there you can experiment with different network architectures, optimizers, or datasets such as Fashion-MNIST or CIFAR-10.

## Next Steps

- Explore data augmentation techniques using `torchvision.transforms`.
- Try using a convolutional neural network (CNN) to improve accuracy.
- Save the trained model and load it later for predictions.

This repository is intentionally kept simple. Feel free to modify the example, add notebooks, or expand it into more complex projects as you learn.

