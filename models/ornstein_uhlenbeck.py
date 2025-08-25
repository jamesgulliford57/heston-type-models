from models.stochastic_model import StochasticModel


class OrnsteinUhlenbeck(StochasticModel):
    """
    Ornstein-Uhlenbeck model describing random evolution of stock price with
    mean-reverting drift and constant volatility.

    dS_t = κ(η - S_t) dt + λ dW_t

    where:
        - S_t = asset price at time t,
        - κ = mean reversion rate,
        - η = long-term mean,
        - λ = volatility,
        - W_t = Brownian motion.
    """
    def __init__(self, **model_params):
        """

        Parameters
        ---
        model_params : dict
            Dictionary containing model parameters.
        """
        state = ['price']
        super().__init__(state=state, drift=self.drift, diffusion=self.diffusion,
                         diffusion_prime=self.diffusion_prime, **model_params)
        if not hasattr(self, 'kappa'):
            raise TypeError('OrnsteinUhlenbeck class cannot be instantiated without mean reversion rate, kappa. '
                            'Please set in model_params in config_file.')
        if not hasattr(self, 'eta'):
            raise TypeError('OrnsteinUhlenbeck class cannot be instantiated without long-term mean, eta. '
                            'Please set in model_params in config_file.')
        if not hasattr(self, 'lmbda'):
            raise TypeError('OrnsteinUhlenbeck class cannot be instantiated without volatility, lmbda. '
                            'Please set in model_params in config_file.')

    def drift(self, price):
        """
        Model drift

        Parameters
        ---
        price : float
            Asset price
        """
        return self.kappa * (self.eta - price)

    def diffusion(self, price):
        """
        Model volatility

        Parameters
        ---
        price : float
            Asset price
        """
        return self.lmbda

    def diffusion_prime(self, price):
        """
        Compute derivative of the model volatility e.g. for use in Milstein scheme.

        Parameters
        ---
        price : float
            Asset price
        """
        return 0
