import numpy as np
from abc import ABCMeta, abstractmethod


class StochasticModel(metaclass=ABCMeta):
    def __init__(self, state, drift, diffusion, diffusion_prime=None, model_params=None):
        """
        Class for general SDEs to produce numerical solutions.

        Parameters
        ----------
        state: list
            Components of the state vector. Converted to np.ndarray.
        drift : float, np.ndarray, or callable
            SDE drift
        diffusion : float, np.ndarray, or callable
            SDE diffusion coefficient
        diffusion_prime : float, np.ndarray, or callable
            Derivative of SDE diffusion coefficient required for some numerical solving schemes e.g. Milstein.
        """
        if model_params is None:
            model_params = {}
        if not isinstance(drift, (float, np.ndarray)) and not callable(drift):
            raise TypeError(f'mu must be a float, np.ndarray, or callable but got {type(drift).__name__}')

        if not isinstance(diffusion, (float, np.ndarray)) and not callable(diffusion):
            raise TypeError(f'sigma must be a float, np.ndarray, or callable but got {type(diffusion).__name__}')

        if not isinstance(model_params, dict):
            raise TypeError(f'model_params must be a dict, np.ndarray, or callable but got '
                            f'{type(model_params).__name__}')
        self.state = np.array(state)
        self.drift = drift
        self.diffusion = diffusion
        if diffusion_prime:  # If model contains derivative of diffusion coefficient
            self.diffusion_prime = diffusion_prime
        self.model_params = model_params | {'state': state}
        for key, value in model_params.items():
            setattr(self, key, value)
        if not hasattr(self, 'risk_free_rate'):
            raise TypeError('StochasticModel class cannot be instantiated without risk_free_rate. '
                            'Please set in model_params in config_file.')

    @abstractmethod
    def drift(self, *args):
        """
        Model drift
        """
        raise NotImplementedError("Drift function not implemented")

    @abstractmethod
    def diffusion(self, *args):
        """
        Model diffusion
        """
        raise NotImplementedError("Diffusion function not implemented")






