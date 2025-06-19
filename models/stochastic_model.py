import numpy as np
from abc import ABCMeta, abstractmethod

class StochasticModel(metaclass=ABCMeta):
    def __init__(self, state, drift, diffusion, diffusion_prime=None, model_params={}):
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
        self.model_params = model_params | {'state' : state}
        for key, value in model_params.items():
            setattr(self, key, value) 

    @abstractmethod
    def _drift(self, *args):
        """
        Model drift
        """
        raise NotImplementedError("Drift function not implemented")
    
    @abstractmethod
    def _diffusion(self, *args):
        """
        Model diffusion
        """
        raise NotImplementedError("Diffusion function not implemented")
    



    

