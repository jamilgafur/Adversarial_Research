from typing import List, Tuple
from utils.Swarm_Observer import Particle
import numpy as np
import torch

def apso(X_train, cost_func, num_particles = 10, max_iter = 10, w = 1.0, c1 = 0.8, c2 = 0.2): 
    """
    Runs the Adversarial Particle Swarm Optimization algorithm.

    Args:
        X_train (List[np.ndarray]): The training data as a list of numpy arrays.
        cost_func (Tuple[torch.nn.Module, callable]): A tuple containing a PyTorch model and a cost function which accepts the model and input data and returns a scalar cost.
        num_particles (int, optional): The number of particles in the swarm. Defaults to 10.
        max_iter (int, optional): The maximum number of iterations to run the algorithm. Defaults to 10.
        w (float, optional): The constant inertia weight. Defaults to 1.0.
        c1 (float, optional): The cognitive constant. Defaults to 0.8.
        c2 (float, optional): The social constant. Defaults to 0.2.

    Returns:
        Tuple[List[np.ndarray], List[float]]: A tuple containing the list of particle positions and the list of best particle errors.

    """
    num_dimensions = X_train[0].flatten().shape[0]

    # establish the swarm
    swarm = []
    for i in range(num_particles):
        x0 = X_train[i].flatten()  # use data as initial positions
        swarm.append(Particle.Particle(x0=x0, num_dimensions=num_dimensions, w=w, c1=c1, c2=c2))

    # initialize best position and error for the group
    pos_best_g = swarm[0].position_i.copy()
    err_best_g = swarm[0].err_i

    # begin optimization loop
    for _ in range(max_iter):
        # evaluate fitness and update the best position and error for the group
        for p in swarm:
            p.evaluate(costFunc=cost_func[1], model=cost_func[0])
            if p.err_i > err_best_g:
                pos_best_g = p.position_i.copy()
                err_best_g = p.err_i

        # update velocities and positions
        for p in swarm:
            p.update_velocity(pos_best_g=pos_best_g)
            p.update_position(bounds=[(0, 1) for _ in range(num_dimensions)])

    # return list of particle positions and best particle errors
    return ([p.position_i for p in swarm], [p.err_best_i for p in swarm])
