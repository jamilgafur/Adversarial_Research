import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import pdb
from CNN import *

def seedEverything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Builds a pytorch CNN and trains it on the MNIST dataset
def main():
    # Set seed for reproducibility
    seedEverything(1234)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 1

    # Load Data
    train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize network
    model = buildCNN(num_classes).to(device)


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    accuracies, losses = train(model, train_loader, criterion, optimizer, num_epochs, device)

    # plot the accuracy and loss for each epoch
    plt.plot(accuracies, label='accuracy')
    plt.plot(losses, label='loss')
    plt.legend()
    plt.savefig('MNIST_CNN_Information.png')

    # Check accuracy on training & test to see how good our model
    testacc = test(model, test_loader, device)
    print(f"Test accuracy: {testacc}")    

    # save the model
    torch.save(model.state_dict(), '../saved_models/MNIST_CNN.pt')
def test(model, test_loader, device):
    """
    Tests the model and returns the accuracy
    :param model: The model to test
    :param test_loader: The data loader for the test data
    :param device: The device to test on
    :return: The accuracy
    """
    # Test Network
    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (data, targets) in enumerate(test_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

        return float(num_correct)/float(num_samples)*100


def train(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    Trains the model and returns the accuracy and loss for each epoch
    :param model: The model to train
    :param train_loader: The data loader for the training data
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param num_epochs: The number of epochs to train for
    :param device: The device to train on
    :return: The accuracy and loss for each epoch
    """
    # Train Network, save and plot the accuracy and loss for each epoch
    accuracies = []
    losses = []
    for epoch in range(num_epochs):
        accuracy =0
        loss = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # save the loss and accuracy for each batch
            loss += loss.item()
            accuracy += (scores.argmax(1) == targets).sum() / float(targets.shape[0])

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
        # normalize the loss and accuracy for each epoch
        loss /= len(train_loader)
        accuracy /= len(train_loader)
        accuracies.append(accuracy.item())
        losses.append(loss.item())
        print(f"Epoch {epoch} loss: {loss} accuracy: {accuracy}")
    return accuracies, losses
    

main()