from models.stochastic_model import StochasticModel


class CoxIngersollRoss(StochasticModel):
    """
    Cox-Ingersoll-Ross model describing random evolution of stock price with mean reverting drift and global
    volatility. Variant of OU model with local coefficient of volatility.

    dS_t = κ(η - S_t) dt + λ√S_t dW_t

    where:
        - S_t = asset price or rate at time t,
        - κ = mean reversion rate,
        - η = long-term mean,
        - λ = volatility coefficient,
        - W_t = Brownian motion.
    """
    def __init__(self, **model_params):
        """
        Parameters
        ----------
        model_params : dict
            Dictionary containing model parameters.
        """
        state = ['price']
        super().__init__(state=state, drift=self.drift, diffusion=self.diffusion,
                         diffusion_prime=self.diffusion_prime, **model_params)
        if not hasattr(self, 'kappa'):
            raise TypeError('CoxIngersollRoss class cannot be instantiated without mean reversion rate, kappa. '
                            'Please set in model_params in config_file.')
        if not hasattr(self, 'eta'):
            raise TypeError('CoxIngersollRoss class cannot be instantiated without long-term mean, eta. '
                            'Please set in model_params in config_file.')
        if not hasattr(self, 'lmbda'):
            raise TypeError('CoxIngersollRoss class cannot be instantiated without volatility coefficient, lmbda. '
                            'Please set in model_params in config_file.')

    def drift(self, price):
        """
        Model drift

        Parameters
        ----------
        price : float
            Asset price
        """
        return self.kappa * (self.eta - price)

    def diffusion(self, price):
        """
        Model volatility

        Parameters
        ----------
        price : float
            Asset price
        """
        return self.lmbda * price ** 0.5

    def diffusion_prime(self, price):
        """
        Compute derivative of the model volatility e.g. for use in Milstein scheme.

        Parameters
        ----------
        price : float
            Asset price
        """
        return 0
