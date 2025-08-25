import numpy as np
from models.stochastic_model import StochasticModel


class Heston(StochasticModel):
    """
    Heston model describing random evolution of stock price and associated stochastic volatility.

    dS_t = r S_t dt + √V_t S_t dW₁_t
    dV_t = λ(σ² - V_t) dt + ξ√V_t dW₂_t
    d⟨W₁, W₂⟩ₜ = ρ dt

    where:
        - S_t = asset price at time t,
        - V_t = instantaneous variance at time t,
        - r = risk-free rate,
        - λ = mean reversion rate,
        - σ² = long-term variance,
        - ξ = volatility of volatility,
        - ρ = correlation between the two Brownian motions W₁ and W₂.
    """
    def __init__(self, **model_params):
        """

        Parameters
        ---
        model_params : dict
            Dictionary containing model parameters.
        """
        state = ['price', 'volatility']
        super().__init__(state=state, drift=self.drift, diffusion=self.diffusion,
                         diffusion_prime=self.diffusion_prime, **model_params)
        if not hasattr(self, 'lmbda'):
            raise TypeError('Heston class cannot be instantiated without mean reversion rate, lmbda. '
                            'Please set in model_params in config_file.')
        if not hasattr(self, 'sigma'):
            raise TypeError('Heston class cannot be instantiated without long-term standard deviation, sigma. '
                            'Please set in model_params in config_file.')
        if not hasattr(self, 'xi'):
            raise TypeError('Heston class cannot be instantiated without volatility of volatility, xi. '
                            'Please set in model_params in config_file.')
        if not hasattr(self, 'rho'):
            raise TypeError('Heston class cannot be instantiated without Brownian motion correlation, rho. '
                            'Please set in model_params in config_file.')

    def drift(self, price, volatility):
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

    def diffusion(self, price, volatility):
        """
        Model volatility

        Parameters
        ---
        price : float
            Asset price
        volatility: float
            Asset volatility
        """
        return np.array([[price * np.sqrt(np.abs(volatility)), 0],
                         [self.rho * self.xi * np.sqrt(np.abs(volatility)),
                          np.sqrt(1 - self.rho ** 2) * self.xi * np.sqrt(np.abs(volatility))]])

    def diffusion_prime(self, price, volatility):
        """
        Compute derivative of the model volatility e.g. for use in Milstein scheme.

        Parameters
        ---
        price : float
            Asset price
        volatility: float
            Asset volatility
        """
        price_derivative = np.array([[np.sqrt(abs(volatility)), 0], [0, 0]])
        volatility_derivative = np.array([[0.5 * price / np.sqrt(abs(volatility)), 0],
                                 [0.5 * self.rho * self.xi / np.sqrt(abs(volatility)),
                                  0.5 * np.sqrt(1-self.rho**2) * self.xi / np.sqrt(abs(volatility))]])

        return price_derivative, volatility_derivative
