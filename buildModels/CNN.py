from torch import nn

def buildCNN(output_size):
    """
    Builds a CNN model
    :param output_size: The number of outputs
    :return: The model
    """
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(7*7*16, output_size)
    )
    