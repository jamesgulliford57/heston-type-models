import numpy as np
import sys
sys.path.append('..')
from utils.data_utils import write_json, write_npy

class StochasticModel:
    def __init__(self, state, drift, diffusion, diffusion_prime=None, **model_params):
        """
        Superclass for general SDEs with built in methods to produce numerical solutions.

        Parameters
        ---
        state: list
            Components of the state vector. Converted to np.ndarray.
        drift : float, np.ndarray, or callable
            SDE drift  
        diffusion : float, np.ndarray, or callable
            SDE diffusion coefficient 
        diffusion_prime : float, np.ndarray, or callable
            Derivative of SDE diffusion coefficient required for some numerical solving schemes e.g. Milstein.
        **model_params : dict
            Model parameters to evaluate mu and sigma and setup. 
        """
        if not isinstance(drift, (float, np.ndarray)) and not callable(drift):
            raise TypeError(f'mu must be a float, np.ndarray, or callable but got {type(drift).__name__}')

        if not isinstance(diffusion, (float, np.ndarray)) and not callable(diffusion):
            raise TypeError(f'sigma must be a float, np.ndarray, or callable but got {type(diffusion).__name__}')

        if not isinstance(model_params, dict):
            raise TypeError(f'model_params must be a dict, np.ndarray, or callable but got {type(model_params).__name__}')
        
        self.state = np.array(state)
        self.drift = drift 
        self.diffusion = diffusion 
        if diffusion_prime: # If model contains derivative of diffusion coefficient
            self.diffusion_prime = diffusion_prime
        # Unpack model parameters
        self.model_params = model_params
        for key, value in model_params.items():
            setattr(self, key, value) 

    def _euler_maryuama_scheme(self, path_samples, n, h, t):
        """
        Simulates one path using the Euler-Maruyama scheme to solve the 
        SDE defined by mu and sigma.

        Parameters
        --- 
        path_samples : dict
            Initial condition and arrays for solution trajectories.
        n : int
            Discretisation parameter, number of intervals to divide time horizon into.
        h : float
            Time step size.
        t : np.ndarray
            Time array to simulate over.
        """
        dim = path_samples.shape[0] # Identify dimension of solution
        bm_samples = np.zeros((dim, n))  # Initialise sample and Brownian motion arrays
        # Simulation loop
        for i in range(1, n):
            bm_step = np.random.normal(0, np.sqrt(h), dim)
            bm_samples[:, i] = bm_samples[:, i - 1] + bm_step
            current_state = path_samples[:, i - 1]
            if dim == 1:
                path_samples[:, i] = current_state + self.drift(current_state) * h + self.diffusion(current_state) * bm_step
            else:
                path_samples[:, i] = current_state + self.drift(*current_state) * h + np.dot(self.diffusion(*current_state), bm_step)
        # Return time and samples arrays
        return t, path_samples
    
    def _milstein_scheme(self, path_samples, n, h, t):
        """
        Simulates one path using the Milstein scheme to solve the SDE defined by mu and sigma.
        
        Parameters
        --- 
        path_samples : dict
            Initial condition and arrays for solution trajectories.
        n : int
            Discretisation parameter, number of intervals to divide time horizon into.
        h : float
            Time step size.
        t : np.ndarray
            Time array to simulate over.
        """
        if not self.diffusion_prime:
            raise ValueError('Diffusion_prime not provided. Derivative of diffusion coefficient is required to simulate Milstein scheme.')
       
        dim = path_samples.shape[0] # Identify dimension of solution
        bm_samples = np.zeros((dim, n))  # Initialise sample and Brownian motion arrays
        # Simulation loop
        for i in range(1, n):
            bm_step = np.random.normal(0, np.sqrt(h), dim)
            bm_samples[:, i] = bm_samples[:, i - 1] + bm_step
            current_state = path_samples[:, i - 1]
            if dim == 1:
                path_samples[:, i] = current_state + self.drift(current_state) * h + self.diffusion(current_state) * bm_step + 0.5 * self.diffusion(current_state) * self.diffusion_prime(current_state) * (bm_step ** 2 - h)
            else:
                path_samples[:, i] = current_state + self.drift(*current_state) * h + np.dot(self.diffusion(*current_state), bm_step) + 0.5 * np.einsum('ij,ijk->ik', self.diffusion(*current_state), self.diffusion_prime(*current_state)) @ (bm_step ** 2 - h)
        # Return time and samples arrays
        return t, path_samples
    
    def simulate_model(self, init_value, final_time, n, num_paths, output_directory, scheme='euler'):
        """
        Simulates numerical solution to SDE defined by mu and sigma using provided numerical scheme.
        
        Parameters
        ---
        init_value : float, or np.ndarray
            Initial condition for random solution trajectories, provided as 1D row array.
        T : float or int 
            Time horizon, simulation runs from t = 0 to t = T.
        n : int 
            Discretisation parameter, number of intervals T is divided into.
        N : int or float
            Number of trajectories to be simulated.
        scheme : str 
            Defines scheme to be use to simulate model.
        """
        # Initialise price and volatility arrays to hold num_samples samples with discretisation parameter n
        samples = {key : np.zeros((num_paths, n)) for key in self.state}
        # Set initial value for each path
        for i, key in enumerate(self.state):
            samples[key][:, 0] = init_value[i]
        # Discretise time interval
        h = final_time / n
        t = np.linspace(0, final_time, n)
        # Select numerical scheme (create list of schemes in another file and use for selection)
        if scheme == 'euler' or scheme == 'Euler':
            for i in range(num_paths):
                if i % 10 == 0 and i > 0:
                    print(f'Simulation {i}/{num_paths} complete')
                path_samples = {key : samples[key][i, :] for key in self.state}
                path_samples = np.vstack([value for value in path_samples.values()])
                t, path_samples = self._euler_maryuama_scheme(path_samples, n, h, t)
                for j, key in enumerate(self.state):
                    samples[key][i, :] = path_samples[j]
        elif scheme == 'milstein' or scheme == 'Milstein':
            for i in range(num_paths):
                if i % 10 == 0 and i > 0:
                    print(f'Simulation {i}/{num_paths} complete')
                path_samples = {key : samples[key][i, :] for key in self.state}
                path_samples = np.vstack([value for value in path_samples.values()])
                t, path_samples = self._milstein_scheme(path_samples, n, h, t)
                for j, key in enumerate(self.state):
                    samples[key][i, :] = path_samples[j]
        else:
            raise ValueError(f'scheme "{scheme}" not recognised')
        # Finish simulation and write data to npy file for analysis
        print(f'{scheme} scheme simulation complete')
        # Write output files
        write_npy(output_directory, t=t, samples=samples) # Samples 
        sim_params = {'init_value' : list(init_value), 'final_time' : final_time, 'n' : n, 'num_paths': num_paths, 'scheme' : scheme}
        params = self.model_params | sim_params
        write_json(output_directory, params=params) # Parameters
