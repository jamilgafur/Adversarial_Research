import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.Attacks import *
from utils.Swarm_Observer.Swarm import * 


def fgsm_generator(X_data, y_data, model, epsilons):
    """
    Generates adversarial examples for the given input data and model for each given epsilon value.

    Args:
        X_data (numpy.ndarray): The input data to be modified.
        y_data (numpy.ndarray): The true labels of the input data.
        model (torch.nn.Module): The neural network model.
        epsilons (list): List of epsilon values to use for generating adversarial examples.

    Returns:
        A dictionary containing a list of tuples for each epsilon value. Each tuple contains the original and adversarial noise for each input data point.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adversarial_points = {}
    for epsilon in epsilons:
        # the list of tuples containing the original and adversarial noise for each input data point and epsilon value
        adv_list = []
        for x, y in zip(X_data, y_data):
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
            perturbed_x = fgsm_attack(x, y, epsilon, model)
            adv_list.append((x.cpu().detach().numpy(), perturbed_x))
            
        adversarial_points[epsilon] = adv_list

    return adversarial_points

def apsoGenerator(X_train, costFunc, epochs=100, xlow=-10, xhigh=10, ylow=0, yhigh=10, type="bin"):
    """
    Generate points using APSO algorithm.

    Args:
        X_train (numpy.ndarray): Input data.
        costFunc (tuple): Tuple containing the model and the cost function.
        epochs (int): Number of epochs for APSO algorithm.
        xlow (float): Minimum x value for the plot.
        xhigh (float): Maximum x value for the plot.
        ylow (float): Minimum y value for the plot.
        yhigh (float): Maximum y value for the plot.
        type (str): Type of problem. Either "bin" or "img".

    Returns:
        List of tuples containing the generated positions and labels.

    """
    os.makedirs("./Output/apso/", exist_ok=True)

    model = costFunc[0]
    positions, labels = apso(X_train, costFunc)

    return (positions, labels)
