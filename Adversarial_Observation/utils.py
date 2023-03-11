from torch import nn
import numpy as np
import torch
from typing import Tuple, List


def cleanPositions(positions: List[torch.Tensor]) -> torch.Tensor:
    """
    This function converts the positions and best positions to a list of tensors, combines all unique positions and best positions
    and removes duplicates. It then adds an extra dimension and returns the positions.
    """
    positions = torch.cat(positions)
    positions = torch.unique(positions, dim=0)
    # add an extra dimension
    positions = torch.unsqueeze(positions, 1)
    return positions

def generate_random_inputs(num_of_inputs: int, input_shape: Tuple[int, int, int], mu: float=0, sigma: float=1) -> np.ndarray:
    """
    This function generates random inputs of a given shape from a normal distribution with mean mu and standard deviation sigma.
    :param num_of_inputs: the number of inputs to generate
    :param input_shape: the shape of the inputs to generate
    :param mu: the mean of the normal distribution
    :param sigma: the standard deviation of the normal distribution
    :return: a numpy array of shape (num_of_inputs, *input_shape) containing the generated inputs
    """
    return np.random.normal(mu, sigma, size=(num_of_inputs, *input_shape))

def buildCNN(output_size: int) -> nn.Sequential:
    """
    Builds a convolutional neural network model.
    :param output_size: The number of output classes.
    :return: The model.
    """
    # Create a sequential container for building the model
    return nn.Sequential(
        # Add a 2D convolutional layer with 1 input channel, 8 output channels, 3x3 kernel size,
        # 1 pixel padding, and 1 pixel stride.
        nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
        # Add a ReLU activation layer.
        nn.ReLU(),
        # Add a 2D max pooling layer with 2x2 kernel size and 2 pixel stride.
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Add another 2D convolutional layer with 8 input channels, 16 output channels, 3x3 kernel size,
        # 1 pixel padding, and 1 pixel stride.
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        # Add another ReLU activation layer.
        nn.ReLU(),
        # Add another 2D max pooling layer with 2x2 kernel size and 2 pixel stride.
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Add a flattening layer to convert the 2D feature maps to a 1D vector.
        nn.Flatten(),
        # Add a fully connected layer with 7x7x16 input features and output_size output features.
        nn.Linear(7*7*16, output_size),
        # Add a softmax activation layer to convert the outputs to probabilities.
        nn.Softmax(dim=1)
    )

def seedEverything(seed: int) -> None:
    """
    Seeds the random number generators for Python, PyTorch, and CUDA to make the results reproducible.
    :param seed: The seed to use for random number generation.
    """
    # Seed NumPy for random number generation.
    np.random.seed(seed)
    # Seed PyTorch for random number generation.
    torch.manual_seed(seed)
    # Seed CUDA for random number generation (if available).
    torch.cuda.manual_seed_all(seed)
