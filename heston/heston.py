import numpy as np
from sde_super_modular import StochasticModel

class HestonModel(StochasticModel):
    def __init__(self, **model_params):
        """
        Heston model is a stochastic model describing the random evolution
        of a stock price and associated volatility
        """
        super().__init__(drift=self._heston_drift, diffusion=self._heston_diffusion, diffusion_prime=self._heston_diffusion_prime, **model_params)
    
    def _heston_drift(self, S, V):
        """
        Drift term for the Heston model
        """
        return np.array([self.r * S, self.lmbda * (self.sigma ** 2 - V)])

    def _heston_diffusion(self, S, V):
        """ 
        Compute Heston model volatility
        """
        return np.array([[S * np.sqrt(np.abs(V)), 0], [self.rho * self.xi * np.sqrt(np.abs(V)), np.sqrt(1 - self.rho ** 2) * self.xi * np.sqrt(np.abs(V))]])
    
    def _heston_diffusion_prime(self, S, V):
        """
        Compute derivative of the Heston model diffusion coefficient e.g. for use in
        Milstein scheme
        """
        derivative_S = np.array([[np.sqrt(V), 0], [0, 0]])
        derivative_V = np.array([[0.5 * S / np.sqrt(V), 0], [0.5 * self.rho * self.xi / np.sqrt(V), 0.5 * np.sqrt(1-self.rho**2) * self.xi / np.sqrt(V)]])
        return derivative_S, derivative_V

#heston1 = HestonModel(r=0.2, lmbda=1.0, sigma=0.5, xi=1.0, rho=-0.5)
#heston1.simulate_model(init_value=np.array([1, 0.16]), T=5, n=10000, N=100, scheme='euler')

