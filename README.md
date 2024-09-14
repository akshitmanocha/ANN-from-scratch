# Feedforward Neural Network from Scratch (MNIST)

This repository contains an implementation of a simple feedforward neural network from scratch using NumPy, designed to classify images from the MNIST dataset. The model uses the ReLU activation function for hidden layers and SoftMax for output classification. The network is optimized using the Adam optimizer and includes learning rate decay for improved performance.

## Features
- **Two-layer neural network**: The model consists of an input layer, a hidden layer, and an output layer.
- **Activation functions**: 
  - **ReLU** for the hidden layer.
  - **SoftMax** for output layer classification.
- **Optimization**: 
  - **Adam Optimizer**: Efficiently minimizes the loss function using a combination of RMSProp and Momentum.
  - **Learning Rate Decay**: Gradually reduces the learning rate during training for smoother convergence.
- **Dataset**: 
  - **MNIST**: A dataset of 28x28 grayscale images representing handwritten digits (0-9).

## Model Architecture
- **Input layer**: 784 nodes (corresponding to 28x28 pixel images).
- **Hidden layer**: 224 nodes (a fully connected layer with ReLU activation).
- **Output layer**: 10 nodes (representing digit classes 0-9 with SoftMax activation).


