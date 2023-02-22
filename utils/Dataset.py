import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset that takes in a list of data and labels, and returns tensor pairs.
    """
    def __init__(self, data=None, labels=None, generate_points=False, min_x=-5, max_x=5, min_y=-5, max_y=25, dimension=1, num_points=1000, costFunc=None):
        """
        Initializes the Dataset with the given data and labels.

        Args:
            data (list): List of data points.
            labels (list): List of labels corresponding to the data points.

        Returns:
            None
        """
        if generate_points:
            data = Dataset.generate_points(min_x, max_x, min_y, max_y, dimension, num_points)
            labels = Dataset.classify_binary_points(data, costFunc)
            
        self.data = data
        self.labels = labels
        
        assert len(self.labels) == len(self.data)

    def __len__(self):
        """
        Returns the length of the dataset.

        Args:
            None

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a pair of tensors corresponding to a data point and its label.

        Args:
            index (int): Index of the data point to be returned.

        Returns:
            tuple: A pair of tensors consisting of the data point and its label.
        """
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32)

    def generate_points(min_x=-5, max_x=5, min_y=-5, max_y=5, dimension=1, num_points=100):
        """
        Generates a set of points within a scale.

        Parameters
        ----------
        min_x : int
            The smallest value that can be sampled in the x coordinate.
            Default value is -5.
        max_x : int
            The highest value that can be sampled in the x coordinate.
            Default value is 20.
        min_y : int
            The smallest value that can be sampled in the y coordinate.
            Default value is -5.
        max_y : int
            The highest value that can be sampled in the y coordinate.
            Default value is 20.
        dimension : int
            The dimension of each input.
            Default value is 1.
        num_points : int
            The number of points to generate.
            Default value is 100.

        Returns
        -------
        np.array
            A num_points x dimension matrix where values are between low and high.
        """
        x = np.linspace(min_x, max_x, int(np.sqrt(num_points)))
        y = np.linspace(min_y, max_y, int(np.sqrt(num_points)))
        X, Y = np.meshgrid(x, y)
        X = X.reshape((np.prod(X.shape),))
        Y = Y.reshape((np.prod(Y.shape),))
        coord = [(a, b) for a, b in zip(X, Y)]
        return np.array(coord)

    def classify_binary_points(data, function):
        """
        Takes generated points and passes them through a function and classifies them as a set of points.
        If a point is above the function, it's labeled 1; otherwise, it's labeled 0.

        Parameters
        ----------
        data : numpy.array
            An array of points.
        function : callable
            A function that takes in an array of values and returns an array of the same shape.

        Returns
        -------
        np.array
            A binary classification of the input data.
        """
        scores = []
        for point in data:
            real = point[:-1].tolist()
            real.append(function(point[:-1])[0])

            distance = np.sum(point) - np.sum(real)
            score = 1 / (1 + np.exp(-distance))
            scores.append(score)
        return np.array(scores)