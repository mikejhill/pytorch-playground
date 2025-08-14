# PyTorch Playground

This repository is a starting point for experimenting with PyTorch and learning how to build machine learning models.

## Project Idea

A great beginner project is to train a neural network that can recognize handwritten digits using the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. The goal is to classify input images of digits (0–9) into the correct category. MNIST is small enough to work on most computers and is commonly used as a first step into deep learning.

### Why MNIST?
- It is well-known and simple, so you can focus on PyTorch basics.
- Each image is only 28×28 pixels in grayscale, making training relatively quick.
- Tutorials and code examples are widely available if you get stuck.

## Project Layout

```
.
├── data/                  # datasets are downloaded here
├── src/
│   └── pytorch_playground/
│       ├── data.py        # helpers for loading datasets
│       ├── model.py       # the SimpleNet definition
│       └── train.py       # training loop for MNIST
├── tests/                 # basic unit tests
├── requirements.txt       # runtime dependencies
├── requirements-dev.txt   # tooling for development
└── README.md
```

## Getting Started

1. **Install Python 3.8 or newer.** Creating a virtual environment is recommended:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

2. **Install dependencies.**

   ```bash
   pip install -r requirements.txt         # runtime packages
   pip install -r requirements-dev.txt     # tools like black, ruff, pytest
   ```

3. **Run the example training script.** This downloads MNIST to the `data/` directory and trains a small neural network:

   ```bash
   python -m pytorch_playground.train
   ```

4. **Run tests** to make sure everything is working:

   ```bash
   pytest
   ```

From here you can experiment with different network architectures, optimizers, or datasets such as Fashion-MNIST or CIFAR-10.

## Next Steps

- Explore data augmentation techniques using `torchvision.transforms`.
- Try using a convolutional neural network (CNN) to improve accuracy.
- Save the trained model and load it later for predictions.

This repository is intentionally kept simple. Feel free to modify the example, add notebooks, or expand it into more complex projects as you learn.
