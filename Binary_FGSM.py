from utils.Dataset import Dataset
from utils.vis import *
from utils.model.DNN import DNN
from utils import Generators
import sklearn.model_selection as sklearn
import pdb
import random

def costFunc(x):
    return x**2

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
    # gets the data for a binary classification problem
    traindataset = Dataset(generate_points=True, num_points=100 , costFunc=costFunc)
    testdataset = Dataset(generate_points=True, num_points=50  , costFunc=costFunc)
    
    
    # make the Output directory if it doesn't exist
    os.makedirs("./Output", exist_ok=True)
    # plots the data
    plt.scatter([i[0] for i in traindataset.data], [i[1] for i in traindataset.data], c=traindataset.labels)
    # plot the cost function
    plt.plot([i for i in range(-5, 5)], [costFunc(i) for i in range(-5, 5)])
    plt.savefig("./Output/Binary_Data.png")
    # creates the model
    dnn = DNN(2, 100,  1)
    print(dnn)

    # trains the model
    optimizer = torch.optim.Adam(dnn.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=32, shuffle=True)

    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model(dnn, optimizer, criterion, trainloader, testloader, epochs)

    # plot the scores of the model
    scores = []
    for point in traindataset.data:
        scores.append(dnn(torch.tensor(point).float().to(device)).item())
    plt.scatter([i[0] for i in traindataset.data], [i[1] for i in traindataset.data], c=scores)
    plt.savefig("./Output/Binary_Scores.png")
    
    
    # attacks the model
    epsilons = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
    # create random points
    starting_points = []
    for i in range(10):
        starting_points.append([random.uniform(-5, 5), random.uniform(-2, 25)])
    # assign a score to them
    starting_scores = []
    for point in starting_points:
        starting_scores.append(dnn(torch.tensor(point).float().to(device)).item())
    # attack the points
    adversarialPoints = Generators.fgsm_generator(starting_points, starting_scores, dnn, epsilons)

    # plots the adversarial points
    for key in adversarialPoints.keys():
        plt.clf()
        
        # generate a bunch of random starting points
        ending_points = [i[1] for i in adversarialPoints[key]]
        
        # plot the starting point colored by the dnn score
        scores = []
        for point in starting_points:
            scores.append(dnn(torch.tensor(point).float().to(device)).item())
        plt.scatter([i[0] for i in starting_points], [i[1] for i in starting_points], c=scores)
        
        # plot the ending point colored by the dnn score
        scores = []
        for point in ending_points:
            scores.append(dnn(torch.tensor(point).float().to(device)).item())
        plt.scatter([i[0] for i in ending_points], [i[1] for i in ending_points], c=scores)

        # plot arrows from starting point to ending point 
        for i in range(len(starting_points)):
            plt.arrow(starting_points[i][0], starting_points[i][1], ending_points[i][0]-starting_points[i][0], ending_points[i][1]-starting_points[i][1], head_width=0.5, head_length=0.5, fc='k', ec='k')
        
        # plot the cost function
        plt.plot([i for i in range(-5, 5)], [costFunc(i) for i in range(-5, 5)])

        plt.savefig(f"./Output/Binary_Fgsm_{key}.png")

import tqdm

def train_model(model, optimizer, criterion, train_loader, test_loader, num_epochs):
    """
    Train the model for a given number of epochs.

    Args:
    model (nn.Module): PyTorch model to be trained
    optimizer (torch.optim.Optimizer): Optimizer to use for training
    criterion (nn.Module): Loss function to use for training
    train_loader (DataLoader): Training data loader
    test_loader (DataLoader): Validation data loader
    num_epochs (int): Number of epochs to train for

    Returns:
    model (nn.Module): Trained PyTorch model
    train_losses (list): List of training losses for each epoch
    val_losses (list): List of validation losses for each epoch
    train_accs (list): List of training accuracies for each epoch
    val_accs (list): List of validation accuracies for each epoch
    """
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss, train_acc = run_epoch(model, train_loader,  criterion, optimizer, training=True)
        val_loss, val_acc = run_epoch(model, train_loader,  criterion, optimizer, training=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)


    return model, train_losses, val_losses, train_accs, val_accs


def run_epoch(model, dataloader, criterion, optimizer, training=True):
    """
    Runs a single epoch of training or evaluation.

    Parameters:
    -----------
    model : nn.Module
        The PyTorch model to train or evaluate.

    dataloader : DataLoader
        The PyTorch DataLoader containing the data.

    criterion : nn.Module
        The loss function used to compute the loss.

    optimizer : torch.optim.Optimizer
        The optimizer used for updating the model parameters.

    training : bool, optional (default=True)
        Whether the model is being trained or evaluated. If True, the model is in training mode.

    Returns:
    --------
    loss : float
        The average loss over the epoch.

    accuracy : float
        The average accuracy over the epoch.
    """
    if training:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    for i, (inputs, labels) in enumerate(dataloader):
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
            model = model.cuda()

        if training:
            optimizer.zero_grad()

        outputs = model(inputs)
        # flatten the labels
        labels = labels.view(-1, 1)
        loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
        epoch_loss += loss.item()

        if training:
            loss.backward()
            optimizer.step()

        correct = torch.round(outputs).eq(torch.round(labels)).sum().item()
        accuracy = correct / inputs.shape[0]
        epoch_accuracy += accuracy

    loss = epoch_loss / len(dataloader)
    accuracy = epoch_accuracy / len(dataloader)

    return loss, accuracy


if __name__ == "__main__":
    main()