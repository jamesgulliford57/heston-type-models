import numpy as np
import sys
sys.path.append('..')
from utils import write_npy, write_json

class StochasticModel:
    def __init__(self, drift, diffusion, diffusion_prime=None, **model_params):
        """
        Superclass for general SDEs with built in methods to produce numerical solutions

        Parameters
        ---
        drift : float, np.ndarray, or callable
            SDE drift  
        diffusion : float, np.ndarray, or callable
            SDE diffusion coefficient 
        diffusion_prime : float, np.ndarray, or callable
            Derivative of SDE diffusion coefficients. Some numerical solving schemes e.g. Milstein 
            require this evaluation at each update step
        **model_params : dict
            Model parameters to evaluate mu and sigma and setup 
        """
        if not isinstance(drift, (float, np.ndarray)) and not callable(drift):
            raise TypeError(f'mu must be a float, np.ndarray, or callable but got {type(drift).__name__}')

        if not isinstance(diffusion, (float, np.ndarray)) and not callable(diffusion):
            raise TypeError(f'sigma must be a float, np.ndarray, or callable but got {type(diffusion).__name__}')

        if not isinstance(model_params, dict):
            raise TypeError(f'model_params must be a dict, np.ndarray, or callable but got {type(model_params).__name__}')

        self.drift = drift
        self.diffusion = diffusion
        if diffusion_prime:
            self.diffusion_prime = diffusion_prime
        # Unpack model parameters
        self.model_params = model_params
        for key, value in model_params.items():
            setattr(self, key, value) 

    def _euler_maryuama_scheme(self, init_value, n, h, t):
        """
        Simulates one path using the Euler-Maruyama scheme to solve the 
        SDE defined by mu and sigma
        """
        dim = len(init_value) # Identify dimension of solution
        samples, bm_samples = np.zeros((dim, n)), np.zeros((dim, n))  # Initialise sample and Brownian motion arrays
        samples[:, 0] = init_value
        # Simulation loop
        for i in range(1, n):
            bm_step = np.random.normal(0, np.sqrt(h), dim)
            bm_samples[:, i] = bm_samples[:, i - 1] + bm_step
            samples[:, i] = samples[:, i - 1] + self.drift(samples[0, i - 1], samples[1, i - 1]) * h + np.dot(self.diffusion(samples[0, i - 1], samples[1, i - 1]), bm_step)
        return t, samples[0], samples[1]
    
    def _milstein_scheme(self, init_value, n, h, t):
        """
        Simulates one path using the Milstein scheme to solve the 
        SDE defined by mu and sigma
        """
        if not self.diffusion_prime:
            raise ValueError('diffusion_prime not provided. Derivative of diffusion coefficient is required to simulate Milstein scheme')
        dim = len(init_value) # Identify dimension of solution
        samples, bm_samples = np.zeros((dim, n)), np.zeros((dim, n))  # Initialise sample and Brownian motion arrays
        samples[:, 0] = init_value
        # Simulation loop
        for i in range(1, n):
            bm_step = np.random.normal(0, np.sqrt(h), dim)
            bm_samples[:, i] = bm_samples[:, i - 1] + bm_step
            samples[0, i] = (samples[0, i - 1] + self.drift(samples[0, i - 1], samples[1, i - 1])[0] * h + self.diffusion(samples[0, i - 1], samples[1, i - 1])[0, 0] * bm_step[0]
            + 0.5 * self.diffusion(samples[0, i - 1], samples[1, i - 1])[0, 0] * self.diffusion_prime(samples[0, i - 1], samples[1, i - 1])[0][0, 0] * (bm_step[0] ** 2 - h))
            samples[1, i] = (samples[1, i - 1] + self.drift(samples[0, i - 1], samples[1, i - 1])[1] * h + self.diffusion(samples[0, i - 1], samples[1, i - 1])[1, 0] * bm_step[1]
            + 0.5 * self.diffusion(samples[0, i - 1], samples[1, i - 1])[1, 0] * self.diffusion_prime(samples[0, i - 1], samples[1, i - 1])[1][1, 0] * (bm_step[1] ** 2 - h))
        return t, samples[0], samples[1]
    
    def simulate_model(self, init_value, T, n, N, output_directory, scheme='euler'):
        """
        Simulates numerical solution to SDE defined by mu and sigma using provided numerical scheme
        
        Parameters
        ---
        init_value : float, or np.ndarray
            Initial condition for random solution trajectories, provided as 1D row array
        T : float or int 
            Time horizon, simulation runs from t = 0 to t = T
        n : int 
            Discretisation parameter, number of intervals T is divided into
        N : int or float
            Number of trajectories to be simulated
        scheme : str 
            Defines scheme to be use to simulate model from 'euler', 'milstein'
        """
        # Initialise price and volatility arrays to hold N samples with parameter n
        all_S = np.zeros((N, n))
        all_V = np.zeros((N, n))
        h = T / n
        t = np.linspace(0, T, n)
        # Select numerical scheme
        if scheme == 'euler' or scheme == 'Euler':
            for i in range(N):
                if i % 10 == 0 and i > 0:
                    print(f'Simulation {i}/{N} complete')
                t, all_S[i], all_V[i] = self._euler_maryuama_scheme(init_value, n, h, t)
        elif scheme == 'milstein' or scheme == 'Milstein':
            for i in range(N):
                if i % 10 == 0 and i > 0:
                    print(f'Simulation {i}/{N} complete')
                t, all_S[i], all_V[i] = self._milstein_scheme(init_value, n, h, t)
        else:
            raise ValueError(f'scheme "{scheme}" not recognised')
        # Finish simulation and write data to npy file for analysis
        print(f'{scheme} scheme simulation complete')
        write_npy(output_directory, t=t, S=all_S, V=all_V)
        sim_params = {'init_value' : list(init_value), 'T' : T, 'n' : n, 'N': N, 'scheme' : scheme}
        params = self.model_params | sim_params
        write_json(output_directory, params=params)
        