import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

def flatten(images: torch.Tensor) -> torch.Tensor:
    """Flatten the images into a 1D tensor."""
    return images.view(images.shape[0], -1)

def extract_pca_features(dataloader: DataLoader, n_components: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features using PCA.
    Args:
        dataloader (DataLoader): PyTorch DataLoader object.
        n_components (int): Number of components to keep in PCA.
    Returns:
        Tuple of numpy arrays containing PCA features and corresponding labels.
    """
    features = None
    labels = []
    for images, target in dataloader:
        images = flatten(images)
        if features is None:
            features = images
        else:
            features = torch.cat((features, images), dim=0)
        labels.append(target)
    features = features.numpy()
    labels = np.concatenate(labels)
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    return pca_features, labels

def main() -> None:
    """Main function to extract PCA features from MNIST dataset and plot them."""
    # Define the data transforms
    transform = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    # Download and load the MNIST dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # Extract PCA features from the training and test sets
    train_pca_features, train_labels = extract_pca_features(trainloader)
    test_pca_features, test_labels = extract_pca_features(testloader)

    # Save PCA features and labels
    os.makedirs('./pca_Data', exist_ok=True)
    np.save('./pca_Data/train_pca_features.npy', train_pca_features)
    np.save('./pca_Data/train_labels.npy', train_labels)
    np.save('./pca_Data/test_pca_features.npy', test_pca_features)
    np.save('./pca_Data/test_labels.npy', test_labels)

    # Load the PCA features and labels
    train_pca_features = np.load('./pca_Data/train_pca_features.npy')
    train_labels = np.load('./pca_Data/train_labels.npy')

    # Plot the PCA features
    plt.figure(figsize=(10, 8))
    plt.scatter(train_pca_features[:, 0], train_pca_features[:, 1], c=train_labels, cmap='tab10')
    plt.scatter(test_pca_features[:, 0], test_pca_features[:, 1], c=test_labels, cmap='tab10')
    plt.colorbar()
    plt.title('MNIST Dataset - PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig("MNIST_PCA.png")

if __name__ == '__main__':
    main()
