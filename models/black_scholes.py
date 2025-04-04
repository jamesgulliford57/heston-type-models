from models.stochastic_model import StochasticModel

class BlackScholes(StochasticModel):
    """
    Heston model describing random evolution of stock price with drift 
    and global volatility.

    Attributes
    ---
    risk_free_rate : float
        Global risk-free interest rate.
    q : float
        Dividend yield.
    sigma : float
        Volatility.
    """
    def __init__(self, model_params):
        """

        Parameters
        ---
        model_params : dict
            Dictionary containing model parameters.
        """
        state = ['price']
        super().__init__(state=state, drift=self._drift, diffusion=self._diffusion, diffusion_prime=self._diffusion_prime, model_params=model_params)

    def _drift(self, price):
        """
        Model drift

        Parameters
        ---
        price : float
            Asset price
        """
        return (self.risk_free_rate - self.q) * price

    def _diffusion(self, price):
        """ 
        Model volatility
        
        Parameters
        ---
        price : float
            Asset price
        """
        return self.sigma * price
    
    def _diffusion_prime(self, price):
        """
        Compute derivative of the model volatility e.g. for use in Milstein scheme.
        
        Parameters
        ---
        price : float
            Asset price
        """
        return self.sigma


