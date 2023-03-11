import imageio
from typing import List
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def bestInSwarm(model, positions, epoch, COI, outdir):
        # ===============Logging Best In Swarm===============================
        # get the best of all the positions
        newpredictions = model(positions.to(torch.float32)).detach().numpy()

        # get best score and image
        # best score is highest value in the 5th column
        best_image_location = np.argmax(newpredictions[:, COI])

        best_score = newpredictions[best_image_location][COI]
        best_image = positions[best_image_location][0]
        
        # plot the best of all the positions
        plt.imshow(best_image, cmap='gray')
        plt.title(f'Best Score: {best_score:.2f}')
        plt.colorbar()
        # check if the directory exists
        os.makedirs('{}apso/images/best'.format(outdir), exist_ok=True)
        plt.savefig(f'{outdir}apso/images/best/epoch_{epoch}.png')
        plt.clf()
        return best_score, newpredictions

def convergeOfSwarm(model, positions, epoch,outdir):
     # ===============Logging Average of Swarm===============================
    # get the average of all the positions
    whole_img = torch.sum(positions, dim=0)/len(positions)
    whole_pred = model(whole_img.to(torch.float32).reshape(1,1,28,28)).detach().numpy()
    whole_label = np.argmax(whole_pred)
    
    # plot the average of all the positions
    plt.imshow(whole_img[0], cmap='gray')
    plt.title(f'Best Score: {whole_pred[0][whole_label]:.2f} \n Prediction: {whole_label}')
    plt.colorbar()
    # check if the directory exists
    os.makedirs('{}apso/images/average'.format(outdir), exist_ok=True)
    plt.savefig(f'{outdir}apso/images/average/epoch_{epoch}.png')
    plt.clf()

    return whole_label

def visualizeGIF(foldername: str, filenames: List[str], output_file: str = 'output.gif') -> None:
    """
    Visualizes the images in the folder as a GIF.
    :param foldername: The folder where the images are stored.
    :param epochs: A list of filenames of the images to include in the GIF.
    :param output_file: The name of the output GIF file (default: 'output.gif').
    """
    # Load the images from the file names.
    images = [imageio.imread(f'{foldername}/{filename}') for filename in filenames]

    # Save the images as a GIF.
    saveLoc = "{}/../{}".format(foldername, output_file)
    print("Saving GIF to {}".format(saveLoc))
    imageio.mimsave(saveLoc, images, duration=.5)
