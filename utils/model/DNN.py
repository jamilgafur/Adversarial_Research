import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(torch.nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=1):
        super(DNN, self).__init__()
        # DNN using nn.sequential and relu activation with three layers for the IRIS dataset
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        

    def forward(self, x):

        x = self.fc1(x)

        # apply a sigmoid to the output
        x = torch.sigmoid(x)
        return x