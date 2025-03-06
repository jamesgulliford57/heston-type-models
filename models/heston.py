import numpy as np
from models.stochastic_model import StochasticModel

class Heston(StochasticModel):
    """
    Heston model describing random evolution of stock price and 
    associated stochastic volatility.
    """
    def __init__(self, **model_params):
        """

        Parameters
        ---
        state : list
            Components of the state vector.
        model_params : dict
            Dictionary containing model parameters.
        self.r : float
            Global risk-free interest rate.
        self.lmbda : float
            Mean reversion rate.
        self.sigma : float
            Long-term volatility.
        self.xi : float
            Volatility of volatility.
        self.rho : float
            Correlation coefficient.
        """
        state = ['S', 'V']
        super().__init__(state=state, drift=self._drift, diffusion=self._diffusion, diffusion_prime=self._diffusion_prime, **model_params)
    
    def _drift(self, S, V):
        """
        Model drift

        Parameters
        ---
        S : float
            Stock price
        V : float
            Volatility
        """
        return np.array([self.r * S, self.lmbda * (self.sigma ** 2 - V)])

    def _diffusion(self, S, V):
        """ 
        Model volatility
        
        Parameters
        ---
        S : float
            Stock price
        V : float
            Volatility
        """
        return np.array([[S * np.sqrt(np.abs(V)), 0], [self.rho * self.xi * np.sqrt(np.abs(V)), np.sqrt(1 - self.rho ** 2) * self.xi * np.sqrt(np.abs(V))]])
    
    def _diffusion_prime(self, S, V):
        """
        Compute derivative of the model volatility e.g. for use in Milstein scheme.
        
        Parameters
        ---
        S : float
            Stock price
        V : float
            Volatility
        """
        derivative_S = np.array([[np.sqrt(abs(V)), 0], [0, 0]])
        derivative_V = np.array([[0.5 * S / np.sqrt(abs(V)), 0], [0.5 * self.rho * self.xi / np.sqrt(abs(V)), 0.5 * np.sqrt(1-self.rho**2) * self.xi / np.sqrt(abs(V))]])
        return derivative_S, derivative_V


