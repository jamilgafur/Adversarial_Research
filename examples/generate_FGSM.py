from Adversarial_Observation.Attacks import fgsm_attack
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
    fgsm = fgsm_attack(img, label,  .2, model)
    confidenceBefore = model(torch.tensor(img).to(torch.float32)).max(1)[0].item()
    confidenceAfter = model(torch.tensor(fgsm).to(torch.float32)).max(1)[0].item()

    # Create the save directory if it doesn't exist
    saveDir = './fgsm/'
    os.makedirs(saveDir, exist_ok=True)

    # save the original
    plt.clf()
    plt.imshow(img.flatten().reshape(28,28), cmap='gray')  
    plt.title('Original: label = {}, Confidence = {}'.format(label, round(confidenceBefore, 4)))
    plt.savefig(os.path.join(saveDir, 'original.png'))
    

    # Save the fgsm map
    plt.clf()
    plt.imshow(fgsm.flatten().reshape(28,28), cmap='gray')
    plt.title('After: label = {}, Confidence = {}'.format(label, round(confidenceAfter,4)))
    plt.savefig(os.path.join(saveDir, 'fgsm.png'))
    plt.close()
    




if "__main__" == __name__:
    main()