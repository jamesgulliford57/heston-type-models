from models.stochastic_model import StochasticModel

class OrnsteinUhlenbeck(StochasticModel):
    """
    Orstein-Uhlenbeck model describing random evolution of stock price with 
    mean reverting drift and global volatility.
    """
    def __init__(self, **model_params):
        """

        Parameters
        ---
        model_params : dict
            Dictionary containing model parameters.
        """
        state = ['S']
        super().__init__(state=state, drift=self._drift, diffusion=self._diffusion, diffusion_prime=self._diffusion_prime, **model_params)

    def _drift(self, S):
        """
        Model drift

        Parameters
        ---
        S : float
            Stock price
        """
        return self.kappa * (self.eta - S)

    def _diffusion(self, S):
        """ 
        Model volatility
        
        Parameters
        ---
        S : float
            Stock price
        """
        return self.lmbda
    
    def _diffusion_prime(self, S):
        """
        Compute derivative of the model volatility e.g. for use in Milstein scheme.
        """
        return 0


