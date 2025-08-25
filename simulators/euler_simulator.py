from simulators.simulator import Simulator
import numpy as np


class EulerSimulator(Simulator):
    """
    Euler simulator for simulating solution to SDE.
    """
    def __init__(self, model, simulator_params):
        """
        Constructor for the EulerSimulator class.

        Parameters
        ----------
        model : StochasticModel
            Model to be simulated.
        simulator_params : dict
            Dictionary containing simulator-specific parameters.
        """
        super().__init__(model=model, simulator_params=simulator_params)

    def sim_path(self, path_samples, discretisation_interval):
        """
        Simulates one path using the Euler-Maruyama scheme.

        Parameters
        ---
        path_samples : dict
            Initial condition and arrays for solution trajectories.
        discretisation_interval : float
            Time step size.
        """
        bm_samples = np.zeros((self.dim, self.discretisation_parameter))
        # Simulation loop
        for path_index in range(1, self.discretisation_parameter):
            bm_step = np.random.normal(0, np.sqrt(discretisation_interval), self.dim)
            bm_samples[:, path_index] = bm_samples[:, path_index - 1] + bm_step
            current_state = path_samples[:, path_index - 1]
            if self.dim == 1:
                path_samples[:, path_index] = (current_state + self.drift(current_state) * discretisation_interval
                                               + self.diffusion(current_state) * bm_step)
            else:
                path_samples[:, path_index] = (current_state + self.drift(*current_state) * discretisation_interval
                                               + np.dot(self.diffusion(*current_state), bm_step))
        return path_samples
