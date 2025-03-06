import numpy as np
from models.stochastic_model import StochasticModel

class BlackScholes(StochasticModel):
    """
    Heston model describing random evolution of stock price with 
    global drift and volatility.
    """
    def __init__(self, **model_params):
        """

        Parameters
        ---
        model_params : dict
            Dictionary containing model parameters.
        """
        state = ['S']
        super().__init__(state=state, drift=self._drift, diffusion=self._diffusion, diffusion_prime=self._diffusion_prime, **model_params)

    def _drift(self, S):
        """
        Model drift

        Parameters
        ---
        S : float
            Stock price
        """
        return (self.r - self.q) * S

    def _diffusion(self, S):
        """ 
        Model volatility
        
        Parameters
        ---
        S : float
            Stock price
        """
        return self.sigma * S
    
    def _diffusion_prime(self):
        """
        Compute derivative of the model volatility e.g. for use in Milstein scheme.
        """
        return self.sigma


