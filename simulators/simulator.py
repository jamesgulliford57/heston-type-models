from abc import ABCMeta, abstractmethod
import numpy as np
from utils.data_utils import write_json, write_npy

class Simulator(metaclass=ABCMeta):
    """
    Abstract base class for all simulators.
    """
    def __init__(self, model, simulator_params):
        """
        Constructor for the Simulator class.

        Parameters
        ---
        simulator_params : dict
            Dictionary containing simulator parameters.
        """
        self.simulator_name = self.__class__.__name__
        self.model = model 
        self.model_name = model.__class__.__name__
        self.model_params = model.model_params
        self.state = model.state
        self.dim = len(self.state)
        self.drift = model.drift
        self.diffusion = model.diffusion
        self.diffusion_prime = model.diffusion_prime
        # Unpack simulator parameters
        for key, value in simulator_params.items():
            setattr(self, key, value)
        
        if isinstance(self.initial_value, float):
            self.initial_value = np.array([self.initial_value])
        elif isinstance(self.initial_value, list):
            self.initial_value = np.array(self.initial_value)

    def sim(self, directory):
        """
        Simulates numerical solution to SDE.

        Parameters
        ---
        directory : str
            Directory to write output files to.
        """
        # Initialise price and volatility arrays to hold num_samples samples with discretisation parameter n
        samples = {state_component : np.zeros((self.number_of_paths, self.discretisation_parameter)) for state_component in self.state}
        # Set initial value for each path
        for i, key in enumerate(self.state):
            samples[key][:, 0] = self.initial_value[i]
        # Discretise time interval
        discretisation_interval = self.final_time / self.discretisation_parameter
        time_values = np.linspace(0, self.final_time, self.discretisation_parameter)
        
        # Simulate paths
        for path in range(self.number_of_paths):
            if path % 10 == 0 and path > 0:
                print(f'Path {path}/{self.number_of_paths} simulated')
            path_samples = {key : samples[key][path, :] for key in self.state}
            path_samples = np.vstack([value for value in path_samples.values()])
            path_samples = self.sim_path(path_samples=path_samples, discretisation_interval=discretisation_interval)
            for j, key in enumerate(self.state):
                samples[key][path, :] = path_samples[j]

        # Finish simulation and write data to npy file for analysis
        print(f'{self.simulator_name} simulation complete')
        # Write output files
        write_npy(directory=directory, samples=samples, time_values=time_values) # Samples 
        output = {key : value for key, value in self.__dict__.items() if isinstance(value, (int, float, list, str, dict))}
        write_json(directory=directory, output=output)  

    @abstractmethod
    def sim_path(self):
        """
        Abstract method for simulating a path.
        """
        pass