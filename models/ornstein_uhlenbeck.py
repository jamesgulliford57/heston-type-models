from models.stochastic_model import StochasticModel

class OrnsteinUhlenbeck(StochasticModel):
    """
    Ornstein-Uhlenbeck model describing random evolution of stock price with 
    mean reverting drift and global volatility.

    Attributes
    ---
    kappa : float
        Mean reversion rate.
    eta : float
        Long-term mean.
    lmbda : float
        Volatility.
    """
    def __init__(self, **model_params):
        """

        Parameters
        ---
        model_params : dict
            Dictionary containing model parameters.
        """
        state = ['price']
        super().__init__(state=state, drift=self._drift, diffusion=self._diffusion, diffusion_prime=self._diffusion_prime, **model_params)

    def _drift(self, price):
        """
        Model drift

        Parameters
        ---
        price : float
            Asset price
        """
        return self.kappa * (self.eta - price)

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


