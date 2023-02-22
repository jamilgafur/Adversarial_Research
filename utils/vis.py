import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_adversarial_bin(plot_array, title, model):
    """
    Plot the scatter plot with arrows between original and adversarial points.

    Args:
    plot_array (list): List of original and adversarial points.
    title (str): Title of the plot.
    model: pytorch model object.

    Returns:
    None.
    """
    if len(plot_array) > 0:
        x_original = [i[0][0] for i in plot_array]
        y_original = [i[0][1] for i in plot_array]

        x_adversarial = [i[1][0] for i in plot_array]
        y_adversarial = [i[1][1] for i in plot_array]
        
        model_out_original = model(torch.tensor([i[0] for i in plot_array], dtype=torch.float32)).flatten().tolist()
        model_out_adversarial = model(torch.tensor([i[1] for i in plot_array], dtype=torch.float32)).flatten().tolist()

        plt.clf()
        plt.scatter(x_original, y_original, c=model_out_original)
        for i in plot_array:
            # start at original point go to adversarial point
            plt.arrow(i[0][0], i[0][1], i[1][0]-i[0][0], i[1][1]-i[0][1], width=0.1)

        plt.scatter(x_adversarial, y_adversarial, c=model_out_adversarial)
        plt.plot([i for i in range(-10, 10)], [i**2 for i in range(-10, 10)])
        plt.colorbar()
        plt.xlim(-5, 5)
        plt.ylim(0, 12.5)
        plt.savefig(f"./Output/fgsm/{title}.png")
        plt.clf()
        plt.cla()
    else:
        print(f"no data for {title}")

def plotData(data, labels, xlow=-10, xhigh=10, ylow=0, yhigh=10, func=None, title=""):
    """
    Plots the given data and labels with an optional function overlay.

    Parameters:
    -----------
    data: numpy array
        Array of data to be plotted.

    labels: numpy array
        Array of labels for the data.

    xlow: int
        Minimum value of the x-axis.

    xhigh: int
        Maximum value of the x-axis.

    ylow: int
        Minimum value of the y-axis.

    yhigh: int
        Maximum value of the y-axis.

    func: function or None
        Optional function to overlay on the plot. If None, no function will be plotted.

    Returns:
    --------
    None
    """
    os.makedirs("./Output", exist_ok=True)
    plt.clf()
    plt.scatter([i[0] for i in data], [i[1] for i in data], c=labels, cmap="viridis")
    if func is not None:
        plt.plot([i for i in range(xlow, xhigh)], [func(i) for i in range(xlow, xhigh)])
    plt.xlim(xlow, xhigh)
    plt.ylim(ylow, yhigh)
    plt.colorbar()
    plt.savefig(title)








