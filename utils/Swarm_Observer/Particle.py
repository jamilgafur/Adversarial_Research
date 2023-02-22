import torch
import numpy as np

class Particle:
    def __init__(self, num_dimensions, x0=None, w=1.0, c1=0.8, c2=0.2):
        """
        Initializes a Particle object.

        Args:
        - num_dimensions (int): the number of dimensions for the particle
        - x0 (list, optional): the initial position of the particle. If not provided, random values are used.
        - w (float, optional): the inertia weight (how much to weigh the previous velocity)
        - c1 (float, optional): the cognitive constant
        - c2 (float, optional): the social constant
        """
        self.dim = num_dimensions

        if x0 is not None:
            self.position_i = x0.copy()
        else:
            self.position_i = [np.random.uniform(-10, 10) for _ in range(self.dim)]

        self.velocity_i = [np.random.uniform(-1, 1) for _ in range(self.dim)]
        self.pos_best_i = self.position_i.copy()   # best position individual
        self.err_best_i = -1   # best error individual
        self.err_i = -1   # error individual

        self.w = w
        self.c1 = c1
        self.c2 = c2

    def evaluate(self, costFunc, model):
        """
        Evaluates the current fitness of the particle.

        Args:
        - costFunc (function): a function that calculates the fitness value of the particle
        - model: the PyTorch model to use for calculating the fitness value
        """
        inputData = torch.tensor(self.position_i, dtype=torch.float32)
        self.err_i = costFunc(model, inputData)

        if self.err_i > self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i.copy()
            self.err_best_i = self.err_i
                    
    def update_velocity(self, pos_best_g):
        """
        Updates the particle velocity based on its own position and the global best position.

        Args:
        - pos_best_g (list): the global best position
        """
        r1 = np.random.random()
        r2 = np.random.random()

        vel_cognitive = self.c1 * r1 * np.add(self.pos_best_i, self.position_i)
        vel_social = self.c2 * r2 * np.subtract(pos_best_g, self.position_i)
        self.velocity_i = np.multiply(self.w, np.add(np.add(self.velocity_i, vel_cognitive), vel_social))

    def update_position(self, bounds):
        """
        Updates the particle position based on its velocity.

        Args:
        - bounds (tuple): a tuple of the lower and upper bounds of the search space
        """
        for i in range(self.dim):
            self.position_i[i] = np.clip(self.position_i[i] + self.velocity_i[i], bounds[0], bounds[1])
