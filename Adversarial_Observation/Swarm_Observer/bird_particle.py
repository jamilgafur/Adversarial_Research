import torch
import numpy as np


class BirdParticle:
    """
    Implements the bird class from the paper
    """

    def __init__(self,
                 position: torch.Tensor,
                 velocity: torch.Tensor,
                 bound: callable = None,
                 memory_weight: float = 1.0,
                 cognitive_weight: float = 0.8,
                 social_weight: float = 0.2):
        """
        Initializes a particle.

        Parameters:
        ----------
        position : tensor list
            The initial position of the particle.
            default: None
        velocity : tensor list
            The initial velocity of the particle.
            default: None
        bound : tensor list
            The boundary of the particle.
            default: None
        memory_weight : float
            The memory weight of the particle.
            default: 1.0
        cognitive_weight : float
            The cognitive weight of the particle.
            default: 0.8
        social_weight : float
            The social weight of the particle.
            default: 0.2
        """
        self.position_i = position

        self.history = [self.position_i]
        self.velocity_i = [velocity]
        self.bound = bound

        self.pos_best_i = self.position_i.detach().clone()   # best position individual
        self.err_best_i = -1   # best error individual
        self.err_i = -1   # error individual

        self.memory_weight = memory_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def evaluate(self,
                 cost_func: callable,
                 model: torch.nn.Module):
        """
        Evaluates the current fitness of the particle.

        Parameters:
        ----------
        cost_func : function
            The cost function to evaluate the fitness of the particle.
        model : nn.Module
            The model to evaluate the fitness of the particle.

        Returns:
        ----------
        None
        """
        self.err_i = cost_func(model, self.position_i)

        if self.err_i > self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i.clone().detach()
            self.err_best_i = self.err_i

    def update_velocity(self, pos_best_g: torch.Tensor):
        """
        Updates the particle velocity based on its own position and the global best position.

        Parameters:
        ----------

        pos_best_g : tensor list
            The global best position of the swarm.

        Returns:
        ----------
        None

        """
        r1 = torch.rand(1)
        r2 = torch.rand(1)

        vel_cognitive = self.cognitive_weight * r1 * \
            torch.add(self.pos_best_i, self.position_i)
        vel_social = self.social_weight * r2 * \
            torch.sub(pos_best_g, self.position_i)
        self.velocity_i = np.multiply(
            self.memory_weight,
            torch.add(
                torch.add(
                    self.velocity_i,
                    vel_cognitive),
                vel_social))

    def update_position(self):
        """
        Updates the particle position based on its velocity.

        Parameters:
        ----------
        None

        Returns:
        ----------
        None

        """
        # update position based on velocity
        self.position_i = self.position_i + self.velocity_i
        # apply bounds
        if self.bound is not None:
            self.position_i = self.bound(self.position_i)

        self.history.append(self.position_i)
