import numpy as np
from models.stochastic_model import StochasticModel

class Heston(StochasticModel):
    """
    Heston model describing random evolution of stock price and 
    associated stochastic volatility.

    Attributes
    ---
    risk_free_rate : float
        Global risk-free interest rate.
    lmbda : float
        Mean reversion rate.
    sigma : float
        Long-term volatility.
    xi : float
        Volatility of volatility.
    rho : float
        Correlation coefficient.
    """
    def __init__(self, **model_params):
        """

        Parameters
        ---
        model_params : dict
            Dictionary containing model parameters.
        """
        state = ['price', 'volatility']
        super().__init__(state=state, drift=self._drift, diffusion=self._diffusion, diffusion_prime=self._diffusion_prime, **model_params)
    
    def _drift(self, price, volatility):
        """
        Model drift

        Parameters
        ---
        price : float
            Asset price
        volatility: float
            Asset volatility
        """
        return np.array([self.risk_free_rate * price, self.lmbda * (self.sigma ** 2 - volatility)])

    def _diffusion(self, price, volatility):
        """ 
        Model volatility
        
        Parameters
        ---
        price : float
            Asset price
        volatility: float
            Asset volatility
        """
        return np.array([[price * np.sqrt(np.abs(volatility)), 0], [self.rho * self.xi * np.sqrt(np.abs(volatility)), np.sqrt(1 - self.rho ** 2) * self.xi * np.sqrt(np.abs(volatility))]])
    
    def _diffusion_prime(self, price, volatility):
        """
        Compute derivative of the model volatility e.g. for use in Milstein scheme.
        
        Parameters
        ---
        price : float
            Asset price
        volatility: float
            Asset volatility
        """
        derivative_S = np.array([[np.sqrt(abs(volatility)), 0], [0, 0]])
        derivative_V = np.array([[0.5 * price / np.sqrt(abs(volatility)), 0], [0.5 * self.rho * self.xi / np.sqrt(abs(volatility)), 0.5 * np.sqrt(1-self.rho**2) * self.xi / np.sqrt(abs(volatility))]])
        return derivative_S, derivative_V


