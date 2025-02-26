import numpy as np
from models.stochastic_model import StochasticModel

class HestonModel(StochasticModel):
    def __init__(self, **model_params):
        """
        Heston model is a stochastic model describing the random evolution
        of a stock price and associated volatility.

        Parameters
        ---
        model_params : dict
            Dictionary containing model parameters.
        """
        super().__init__(drift=self._heston_drift, diffusion=self._heston_diffusion, diffusion_prime=self._heston_diffusion_prime, **model_params)
    
    def _heston_drift(self, S, V):
        """
        Drift term for the Heston model.

        Parameters
        ---
        S : float
            Stock price
        V : float
            Volatility
        """
        return np.array([self.r * S, self.lmbda * (self.sigma ** 2 - V)])

    def _heston_diffusion(self, S, V):
        """ 
        Compute Heston model volatility.
        
        Parameters
        ---
        S : float
            Stock price
        V : float
            Volatility
        """
        return np.array([[S * np.sqrt(np.abs(V)), 0], [self.rho * self.xi * np.sqrt(np.abs(V)), np.sqrt(1 - self.rho ** 2) * self.xi * np.sqrt(np.abs(V))]])
    
    def _heston_diffusion_prime(self, S, V):
        """
        Compute derivative of the Heston model diffusion coefficient e.g. for use in
        Milstein scheme.
        
        Parameters
        ---
        S : float
            Stock price
        V : float
            Volatility
        """
        derivative_S = np.array([[np.sqrt(V), 0], [0, 0]])
        derivative_V = np.array([[0.5 * S / np.sqrt(V), 0], [0.5 * self.rho * self.xi / np.sqrt(V), 0.5 * np.sqrt(1-self.rho**2) * self.xi / np.sqrt(V)]])
        return derivative_S, derivative_V


