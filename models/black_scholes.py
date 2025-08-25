from models.stochastic_model import StochasticModel


class BlackScholes(StochasticModel):
    """
    Black-Scholes model describing random evolution of stock price according to a geometric
    Brownian motion i.e. with global drift and global volatility.

    dS_t = (r - q) S_t dt + σ S_t dW_t

    where:
        - S_t = asset price at time t,
        - r = risk-free rate,
        - q = continuous dividend yield,
        - σ = volatility,
        - W_t = Brownian motion.
    """
    def __init__(self, model_params):
        """

        Parameters
        ----------
        model_params : dict
            Dictionary containing model parameters.
        """
        state = ['price']
        super().__init__(state=state, drift=self.drift, diffusion=self.diffusion,
                         diffusion_prime=self.diffusion_prime, model_params=model_params)
        if not hasattr(self, 'q'):
            raise TypeError('BlackScholes class cannot be instantiated without continuous dividend yield, q. '
                            'Please set in model_params in config_file.')
        if not hasattr(self, 'sigma'):
            raise TypeError('BlackScholes class cannot be instantiated without volatility, sigma. '
                            'Please set in model_params in config_file.')

    def drift(self, price):
        """
        Model drift

        Parameters
        ----------
        price : float
            Asset price
        """
        return (self.risk_free_rate - self.q) * price

    def diffusion(self, price):
        """
        Model volatility

        Parameters
        ----------
        price : float
            Asset price
        """
        return self.sigma * price

    def diffusion_prime(self, price):
        """
        Compute derivative of the model volatility e.g. for use in Milstein scheme.

        Parameters
        ----------
        price : float
            Asset price
        """
        return self.sigma


