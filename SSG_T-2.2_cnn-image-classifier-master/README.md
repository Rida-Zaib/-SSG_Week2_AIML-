# Task 2.2 — CNN Image Classifier

## Overview
Trains a convolutional neural network in PyTorch to classify handwritten digit
images, using the same digits dataset as task 2.1 but reshaped back into actual
8x8 images so the CNN can learn spatial patterns instead of a flat feature vector.

## What the notebook does
1. Reshapes the digit vectors into 8x8x1 images and normalizes pixel values
2. Displays a sample of training images with their labels
3. Defines `SimpleCNN` — two convolutional layers with max pooling, followed by
   fully connected layers
4. Trains for 15 epochs with Adam and cross entropy loss
5. Reports test accuracy and saves the trained model (`cnn_model.pt`)

## Files
- `cnn_image_classifier.ipynb` — the full notebook, code + outputs

## How to run
```bash
pip install torch scikit-learn matplotlib
jupyter notebook cnn_image_classifier.ipynb
```
Then Run All Cells.

## Deliverable
Trained CNN model + accuracy report, as required by the Skill Set Go EduTech
AI/ML track, Week 2, Task 2.2.
