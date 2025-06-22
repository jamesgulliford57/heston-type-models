from models.stochastic_model import StochasticModel

class OrnsteinUhlenbeck(StochasticModel):
    """
    Ornstein-Uhlenbeck model describing random evolution of stock price with
    mean-reverting drift and constant volatility.

    dS_t = κ(η - S_t) dt + λ dW_t

    where:
        - S_t is the asset price at time t,
        - κ is the mean reversion rate,
        - η is the long-term mean,
        - λ is the volatility,
        - W_t is a standard Brownian motion.
    """
    def __init__(self, **model_params):
        """

        Parameters
        ---
        model_params : dict
            Dictionary containing model parameters.
        """
        state = ['price']
        super().__init__(state=state, drift=self._drift, diffusion=self._diffusion,
                         diffusion_prime=self._diffusion_prime, **model_params)

    def _drift(self, price):
        """
        Model drift

        Parameters
        ---
        price : float
            Asset price
        """
        return self.kappa * (self.eta - abs(price))

    def _diffusion(self, price):
        """
        Model volatility

        Parameters
        ---
        price : float
            Asset price
        """
        return self.lmbda

    def _diffusion_prime(self, price):
        """
        Compute derivative of the model volatility e.g. for use in Milstein scheme.

        Parameters
        ---
        price : float
            Asset price
        """
        return 0
