from simulators.simulator import Simulator
import numpy as np


class MilsteinSimulator(Simulator):
    """
    Milstein simulator for simulating stochastic processes.
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
        if not self.diffusion_prime:
            raise ValueError("Diffusion_prime not provided. Derivative of diffusion coefficient is "
                             "required to simulate Milstein scheme.")

    def sim_path(self, path_samples, discretisation_interval):
        """
        Simulates one path using the Milstein scheme.

        Parameters
        ----------
        path_samples : dict
            Initial condition and arrays for solution trajectories.
        discretisation_interval : float
            Time step size.
        """
        bm_samples = np.zeros((self.dim, self.discretisation_parameter))
        for i in range(1, self.discretisation_parameter):
            bm_step = np.random.normal(0, np.sqrt(discretisation_interval), self.dim)
            bm_samples[:, i] = bm_samples[:, i - 1] + bm_step
            current_state = path_samples[:, i - 1]
            if self.dim == 1:
                path_samples[:, i] = (current_state + self.drift(current_state) * discretisation_interval
                                      + self.diffusion(current_state) * bm_step
                                      + 0.5 * self.diffusion(current_state) * self.diffusion_prime(current_state)
                                      * (bm_step ** 2 - discretisation_interval))
            else:
                path_samples[:, i] = (current_state + self.drift(*current_state) * discretisation_interval
                                      + np.dot(self.diffusion(*current_state), bm_step)
                                      + 0.5 * np.einsum('ij,ijk->ik', self.diffusion(*current_state),
                                                        self.diffusion_prime(*current_state))
                                      @ (bm_step ** 2 - discretisation_interval))
        return path_samples
