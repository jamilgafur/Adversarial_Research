import torch
import numpy as np

class BirdParticle:
    def __init__(self, position, w=1.0, c1=0.8, c2=0.2):
        """
        Initializes a particle.
        :param position: the initial position of the particle
        :param w: the inertia weight
        :param c1: the cognitive weight
        :param c2: the social weight
        """
        self.position_i = position

        self.history = [self.position_i]
        self.velocity_i = torch.rand(self.position_i.shape) 
        # copy the current position to the best position
        
        self.pos_best_i = self.position_i.detach().clone()   # best position individual
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
        self.err_i = costFunc(model, self.position_i)

        if self.err_i > self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i.clone().detach()
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

    def update_position(self):
        """
        Updates the particle position based on its velocity.
        """
        # update position based on velocity
        self.position_i = self.position_i +  self.velocity_i
        #clip between -1 and 1
        self.position_i = torch.clamp(self.position_i, 0, 1)
        self.history.append(self.position_i)
