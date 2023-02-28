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
from sklearn.cluster import AgglomerativeClustering

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
    # make the directory for the images
    if not os.path.exists('./APSO'):
        os.makedirs('./APSO')

    seedEverything(100)
    # Define training parameters
    batch_size = 1024
    num_epochs = 1
    learning_rate = 0.001

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./MNIST_DATA', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./MNIST_DATA', train=False, transform=transforms.ToTensor(), download=True)

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
    train_ones_dataset = [i for i, l in enumerate(train_dataset.targets) if l == 1]
    train_ones = [train_dataset.data[i].cpu().detach().numpy().reshape(1,1 ,28,28) for i in train_ones_dataset]

    # create a directory if it doesnt exist
    if not os.path.exists('./APSO/ones'):
        os.makedirs('./APSO/ones')

    # plot all of the ones
    for i in range(len(train_ones)):
        plt.imshow(train_ones[i].reshape(28,28), cmap='gray')
        plt.savefig('./APSO/ones/_{}.png'.format(i))
        plt.clf()

    
    epsilons = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]

    # attack the ones points 
    point, labels = Generators.apsoGenerator(train_ones, [model, costFunc], epochs=1000)
    
    # cluster the points
    clusterPoints(point, 2, 'ones')

    # generate random points
    numPoints = 100
    test_ones = [np.random.rand(1,1 ,28,28) for i in range(numPoints)]

    # attack the random points
    point, labels = Generators.apsoGenerator(test_ones, [model, costFunc], epochs=1000)

    # cluster the points
    clusterPoints(point, 2, 'random')

def clusterPoints(point, clusters = 2, title = ''):
    # cluster the points
    clustering = AgglomerativeClustering().fit(point)
    labels = clustering.labels_
    
    # go through each cluster and sum the points
    cluster_0 = []
    cluster_1 = []
    for i in range(len(labels)):
        if labels[i] == 0:
            cluster_0.append(point[i])
        else:
            cluster_1.append(point[i])
    
    # plot the sum of the cluser and it
    plt.imshow(np.sum(cluster_0, 0).reshape(28,28), cmap='gray')
    plt.title('Cluster 0')
    plt.savefig('./APSO/{}_cluster_0.png'.format(title))
    plt.clf()

    plt.imshow(np.sum(cluster_1, 0).reshape(28,28), cmap='gray')
    plt.title('Cluster 1')
    plt.savefig('./APSO/{}_cluster_1.png'.format(title))
    plt.clf()

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