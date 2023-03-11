from Adversarial_Observation.Attacks import saliency_map
import torch
from Adversarial_Observation.utils import buildCNN
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor
import os
import matplotlib.pyplot as plt


def main():
    # Dowload the MNIST dataset
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    mnist = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)

    # Get the first image in the dataset
    img, label = mnist[0]
    
    # Load the model
    model = buildCNN(10)
    model.load_state_dict(torch.load('./saved_models/MNIST_CNN.pt'))
    
    # add the batch size dimension
    img = img.reshape(1,1,28,28)

    # Get the saliency map for the image
    saliency = saliency_map(img, model)

    # Create the save directory if it doesn't exist
    saveDir = './saliency_maps/'
    os.makedirs(saveDir, exist_ok=True)

    # Save the saliency map
    plt.imshow(saliency.flatten().reshape(28,28), cmap='gray')
    plt.title('Saliency Map: label = {}'.format(label))
    plt.savefig(os.path.join(saveDir, 'saliency_map.png'))
    




if "__main__" == __name__:
    main()