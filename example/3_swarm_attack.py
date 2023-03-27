from Adversarial_Observation.Swarm_Observer.Swarm import PSO
from Adversarial_Observation.utils import seedEverything, buildCNN
import matplotlib.pyplot as plt
import torch
import numpy as np
import tqdm 
import pickle
import os
from sklearn.decomposition import PCA

# ==== Global Variables ====
# The value of the column we want to optimize for
endValue: int = 1
#================================================

def costFunc(model, input):
    """
    This function takes a model and a tensor input, reshapes the tensor to be a 28 x 28 image of batch size 1 and passes that through
    the model. Then, it returns the output of the column of interest (endValue) as a numpy array.
    """
    input = torch.reshape(input, (1, 1, 28, 28)).to(torch.float32)
    out = model(input)
    return out.detach().numpy()[0][endValue]


def randomPSO(model, inputs, costFunc, epochs):        
    swarm = PSO(inputs, costFunc, model, w=.8, c1=.5, c2=.5)
    pred =  model(torch.tensor(swarm.pos_best_g.unsqueeze(0)).to(torch.float32))
    best = np.argmax(pred.detach().numpy())
    print("initial best: {} confidence: {}".format(best, pred.detach().numpy()[0][best]))
          
    for i in tqdm.tqdm(range(epochs)):
        swarm.step()
        pred =  model(torch.tensor(swarm.pos_best_g.unsqueeze(0)).to(torch.float32))
        best = np.argmax(pred.detach().numpy())
        print("epoch: {} best: {} confidence: {}".format(i, best, pred.detach().numpy()[0][best]))
          
    

def randomVisualize(model, inputs, costFunc, epochs):
    swarm = PSO(inputs, costFunc, model, w=.8, c1=.5, c2=.5)

    pred =  model(torch.tensor(swarm.pos_best_g.unsqueeze(0)).to(torch.float32))
    best = np.argmax(pred.detach().numpy())
    print("initial best: {} confidence: {}".format(best, pred.detach().numpy()[0][best]))

    # load the PCA data
    with open('./artifacts/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
        

    # load the train reduced data
    train_data_reduced = np.load(open('./artifacts/train_data_reduced.npy', 'rb'))
    train_labels_reduced = np.load(open('./artifacts/train_labels.npy', 'rb'))

    # visualize the swarm in PCA space
    os.makedirs('./artifacts/Random Swarm', exist_ok=True)
    visualizeSwarm(np.array([i.position_i.numpy() for i in swarm.swarm]), train_data_reduced, train_labels_reduced, pca, "./artifacts/Random Swarm/epoch_{}".format(0))

    for i in tqdm.tqdm(range(epochs)):
        swarm.step()
        # visualize the swarm in PCA space
        visualizeSwarm(np.array([i.position_i.numpy() for i in swarm.swarm]), train_data_reduced, train_labels_reduced, pca, "./artifacts/Random Swarm/epoch_{}".format(i+1))
        pred =  model(torch.tensor(swarm.pos_best_g.unsqueeze(0)).to(torch.float32))
        best = np.argmax(pred.detach().numpy())
        print("epoch {} best: {} confidence: {}".format(i, best, pred.detach().numpy()[0][best]))

def visualizeSwarm(positions, stable, stable_lables,  pca, title):
    """
    This function takes a swarm and plots it in PCA space.
    """
    # plot the swarm in PCA space
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    # plot the stable data
    ax.scatter(stable[:,0], stable[:,1], c=stable_lables, cmap='tab10', alpha=.5)
    positions = pca.transform(positions.reshape(-1, 28*28))
    # plot the swarm
    ax.scatter(positions[:,0], positions[:,1], c='black', alpha=.5, marker='x')

    plt.savefig(title)
    

def main():
    seedEverything(44)
    model = buildCNN(10)

    model.load_state_dict(torch.load('./artifacts/mnist_cnn.pt'))
    model.eval()

    points = 500
    input_shape = (points, 1, 28, 28)
    epochs = 100


    random_inputs = np.random.rand(*input_shape)
    # randomPSO(model, random_inputs, costFunc, epochs)

    randomVisualize(model, random_inputs, costFunc, epochs)

if __name__ == "__main__":
    main()