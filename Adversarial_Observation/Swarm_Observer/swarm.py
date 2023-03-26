import torch
import numpy as np
from . import BirdParticle
import tqdm
import pandas as pd


class PSO:
    def __init__(
            self,
            starting_positions: torch.Tensor,
            velocity: torch.Tensor,
            cost_func: callable,
            model: torch.nn.Module,
            bound: callable = None,
            memory_weight: float = 1.0,
            cognitive_weight: float = 0.8,
            social_weight: float = 0.2):
        """
        Initializes the swarm.

        Parameters:
        ----------
        starting_positions : tensor list
            The initial positions of the particles.
            default: None
        velocity : tensor list
            The initial velocities of the particles.
            default: None
        cost_func : callable
            The cost function to be minimized.
            default: None
        model : torch.nn.Module
            The model to be trained.
            default: None
        bound : tensor list
            The boundary of the particles.
            default: None
        memory_weight : float
            The memory weight of the particles.
            default: 1.0
        cognitive_weight : float
            The cognitive weight of the particles.
            default: 0.8
        social_weight : float
            The social weight of the particles.
            default: 0.2

        """
        self.swarm = [
            BirdParticle(
                position=starting_positions[i],
                velocity=velocity[i],
                bound=bound,
                memory_weight=memory_weight,
                cognitive_weight=cognitive_weight,
                social_weight=social_weight) for i in range(
                starting_positions.shape[0])]

        self.cost_func = cost_func
        self.model = model

        # Initialize the best position and error for the group (does not matter
        # for the zeros iteration)
        self.pos_best_g = self.swarm[0].position_i.clone().detach()
        self.err_best_g = self.swarm[0].err_i

    def step(self) -> tuple:
        """
        Performs a single step of the swarm.

        Parameters:
        ----------
        None

        Returns:
        -------
        None

        """
        # Evaluate fitness and update the best position and error for the
        # group.
        for p in self.swarm:
            p.evaluate(self.cost_func, self.model)
            if p.err_i > self.err_best_g:
                self.pos_best_g = p.position_i.clone().detach()
                self.err_best_g = p.err_i

        # Update velocities and positions.
        for p in self.swarm:
            p.update_velocity(pos_best_g=self.pos_best_g)
            p.update_position()

    def get_positions(self) -> torch.Tensor:
        """
        Returns the current positions of each particle in the swarm.

        Parameters:
        ----------
        None

        Returns:
        -------
        positions : torch.Tensor
            The current positions of each particle in the swarm.
        """
        return torch.stack([p.position_i for p in self.swarm])

    def get_best_position(self) -> torch.Tensor:
        """
        Returns the best position of each particle in the swarm.

        Parameters:
        ----------
        None

        Returns:
        -------
        best_positions : torch.Tensor
            The best positions of each particle in the swarm.
        """
        return torch.stack([p.pos_best_i for p in self.swarm])

    def save_history(self, filename: str):
        """
        Saves the history of the swarm to a csv file.

        Parameters:
        ----------
        filename : str
            The name of the file to save the history to.

        Returns:
        -------
        None

        """
        data = [[epoch]+ [i for i in particle.history[epoch].detach().numpy()]
                for particle in self.swarm for epoch in range(len(particle.history))]

        df = pd.DataFrame(data, columns=[
                          'Epoch'] + [f"pos_{i}" for i in range(self.swarm[0].position_i.shape[0])])

        df = df.sort_values(by=['Epoch'])
        df.to_csv(filename, index=False)
