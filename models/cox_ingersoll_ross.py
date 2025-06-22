from models.stochastic_model import StochasticModel

class CoxIngersollRoss(StochasticModel):
    """
    Cox-Ingersoll-Ross model describing random evolution of stock price with mean reverting drift and global
    volatility. Differentiated from normal OU model by the inclusion of sqrt(price) coefficient of volatility.

    dS_t = κ(η - S_t) dt + λ√S_t dW_t

    where:
        - S_t is the asset price or rate at time t,
        - κ is the mean reversion rate,
        - η is the long-term mean,
        - λ is the volatility coefficient,
        - W_t is a standard Brownian motion.
    """
    def __init__(self, **model_params):
        """
        Parameters
        ----------
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
        ----------
        price : float
            Asset price
        """
        return self.kappa * (self.eta - abs(price))

    def _diffusion(self, price):
        """
        Model volatility

        Parameters
        ----------
        price : float
            Asset price
        """
        return self.lmbda * (abs(price) ** 0.5)

    def _diffusion_prime(self, price):
        """
        Compute derivative of the model volatility e.g. for use in Milstein scheme.

        Parameters
        ----------
        price : float
            Asset price
        """
        return 0
