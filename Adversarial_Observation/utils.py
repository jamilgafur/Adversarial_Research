from torch import nn
import numpy as np
import torch
from typing import *


def generate_random_inputs(num_of_inputs: int,
                           input_shape: Tuple,
                           sparcity: float,
                           mu: float = 0,
                           sigma: float = 1,
                           ) -> np.ndarray:
    """
    This function generates random inputs of a given shape from a normal distribution with mean mu and standard deviation sigma.

    Parameters
    ----------
    num_of_inputs : int
        The number of inputs to generate.
    input_shape : Tuple
        The shape of the inputs to generate.
    sparcity : float
        The sparcity of the inputs.
    mu : float
        The mean of the normal distribution.
        default: 0
    sigma : float
        The standard deviation of the normal distribution.
        default: 1
    """
    dense_input = np.random.normal(
        mu, sigma, size=(
            num_of_inputs, *input_shape))
    return np.multiply(dense_input, np.random.choice(
        [0, 1], size=dense_input.shape, p=[sparcity, 1 - sparcity]))


def build_cnn(output_size: int = 10) -> nn.Sequential:
    """
    This function builds a simple CNN with 2 convolutional layers and 2 max pooling layers.

    Parameters
    ----------
    output_size : int
        The number of output classes.
        default: 10

    Returns
    -------
    nn.Sequential
        The CNN model.


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
        nn.Linear(7 * 7 * 16, output_size),
        # Add a softmax activation layer to convert the outputs to probabilities.
        nn.Softmax(dim=1)
    )


def seedEverything(seed: int = 42) -> None:
    """
    This function seeds all the random number generators.

    Parameters
    ----------
    seed : int
        The seed to use.
        default: 42
    """
    # Seed NumPy for random number generation.
    np.random.seed(seed)
    # Seed PyTorch for random number generation.
    torch.manual_seed(seed)
    # Seed CUDA for random number generation (if available).
    torch.cuda.manual_seed_all(seed)
