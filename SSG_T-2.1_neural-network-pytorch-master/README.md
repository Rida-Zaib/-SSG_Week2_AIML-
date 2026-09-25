# Task 2.1 — Neural Network in PyTorch

## Overview
Implements a feed-forward neural network in PyTorch, covering tensors, a custom
Dataset class, DataLoader, and a loss function, trained on the scikit-learn digits
dataset (1,797 8x8 handwritten digit images, flattened to 64 features).

## What the notebook does
1. Loads the digits dataset and splits it into train/test sets
2. Scales the features with `StandardScaler`
3. Wraps the data in a custom `DigitsDataset` (PyTorch `Dataset`) and `DataLoader`
4. Defines a `FeedForwardNet` with two hidden layers (64 -> 32 -> 32 -> 10) and
   ReLU activations
5. Trains for 10 epochs using the Adam optimizer and cross entropy loss
6. Evaluates test accuracy and saves the trained model (`feedforward_model.pt`)

## Files
- `neural_network_pytorch.ipynb` — the full notebook, code + outputs
- `train_nn.py` — the same logic as a standalone script

## How to run
```bash
pip install torch scikit-learn
jupyter notebook neural_network_pytorch.ipynb
```
Then Run All Cells. Or run the script directly: `python3 train_nn.py`

## Deliverable
PyTorch training notebook, as required by the Skill Set Go EduTech AI/ML track,
Week 2, Task 2.1.
