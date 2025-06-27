import numpy as np
from abc import ABCMeta, abstractmethod
from utils.data_utils import write_json, write_npy
from utils.sim_utils import timer

class Simulator(metaclass=ABCMeta):
    """
    Base class for simulators.
    """
    def __init__(self, model, simulator_params):
        """
        Constructor for the Simulator class.

        Parameters
        ---
        model : StochasticModel
            Model to be simulated.
        simulator_params : dict
            Dictionary of simulator parameters.
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
        for key, value in simulator_params.items():
            setattr(self, key, value)
        self.initial_value = np.atleast_1d(self.initial_value)

    @timer
    def sim(self, directory):
        """
        Simulates numerical solution to SDE.

        Parameters
        ----------
        directory : str
            Output directory to write to.
        """
        # Setup paths and discretise interval
        discretisation_interval = self.final_time / self.discretisation_parameter
        time_values = np.linspace(0, self.final_time, self.discretisation_parameter)
        samples = {'time' : time_values} | {state_component :
            np.zeros((self.number_of_paths, self.discretisation_parameter)) for state_component in self.state}
        for path_index, state_component in enumerate(self.state):
            samples[state_component][:, 0] = self.initial_value[path_index]
        # Simulate paths
        for path in range(self.number_of_paths):
            if path > 0 and path % (self.number_of_paths // 10) == 0:
                print(f'Path {path}/{self.number_of_paths} simulated.')
            path_samples = {state_component : samples[state_component][path, :] for state_component in self.state}
            path_samples = np.vstack([value for value in path_samples.values()])
            path_samples = self.sim_path(path_samples=path_samples, discretisation_interval=discretisation_interval)
            path_samples = np.clip(path_samples, a_min=0, a_max=None)  # Ensure non-negativity
            for component_index, state_component in enumerate(self.state):
                samples[state_component][path, :] = path_samples[component_index]
        # Write outputs
        write_npy(directory=directory, samples=samples)
        self.initial_value = self.initial_value.tolist()  # Convert to list for JSON serialization
        params = {key : value for key, value in self.__dict__.items() if
                  isinstance(value, (int, float, list, str, dict))}
        write_json(directory=directory, params=params)

    @abstractmethod
    def sim_path(self):
        """
        Abstract method for simulating a single path.
        """
        pass
