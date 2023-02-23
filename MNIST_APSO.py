import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.model.CNN import CNN
from utils import Generators
import pdb
import numpy as np
import random
import os
import matplotlib.pyplot as plt



def costFunc(model, input):
    # Given the input, reshape it to be a 28 x 28 image of batch size 1 and pass that through the model
    # we are looking for the output of the images that are 2
    input = torch.reshape(input, (1,1,28,28))
    out = model(input)
    # interested in outputs of 2
    return out.detach().numpy()[0][2]

def seedEverything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seedEverything(100)
    # Define training parameters
    batch_size = 1024
    num_epochs = 0
    learning_rate = 0.001

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    # Create data loaders for the training and testing datasets
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the CNN model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs)

    # Test the model
    test(model, test_loader)

    # get all the 1's from the test set
    test_ones_positions = [i for i, l in enumerate(test_dataset.targets) if l == 1]
    test_ones = [test_dataset.data[i].cpu().detach().numpy().reshape(1,1 ,28,28) for i in test_ones_positions]
    
    epsilons = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
    # attack the ones points 
    point, labels = Generators.apsoGenerator(test_ones, [model, costFunc])
    pdb.set_trace()
    quit()
    # make MNIST_FGSM_adversarial_points folder
    os.makedirs('MNIST_FGSM_adversarial_points', exist_ok=True)
    # create a dictionary to store the points based on the label
    points = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    # add the points to the dictionary
    for i in range(len(point)):
        points[labels[i]].append(point[i])
    print("done")
            

def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_images = 0
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

        # Calculate accuracy on training set
        accuracy = 100 * total_correct / total_images
        print('Epoch [{}/{}], Training Accuracy: {:.2f}%'.format(epoch+1, num_epochs, accuracy))

    print('Training finished')

def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
 
main()