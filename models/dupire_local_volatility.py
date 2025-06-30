from models.stochastic_model import StochasticModel
import numpy as np 
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
import models.black_scholes
from sample_analysis.interp_local_volatility import build_local_vol_surface_from_implied

class DupireLocalVolatility(StochasticModel):
    """

    The Dupire model is a way to describe asset price dynamics where volatility is a deterministic function of the asset price and time:

    dS_t = = rS_t dt + σ(S_t,t)S_t dW_t

    S_t = asset price at time 

    𝜎(𝑆,𝑡) = local volatility, a function of both price and time

    r = risk-free rate

    The key idea is: volatility changes over time and with asset level — unlike Black-Scholes which assumes constant volatility.
    """

    def __init__(self,**model_params):
        """
        
        Parameters
        ----------

        model_params: dict 
            Dictionary containing model parameters.

        """
        state = ['price']
        super().__init__(state=state, drift=self._drift, diffusion=self._diffusion, diffusion_prime=self._diffusion_prime, model_params=model_params)

        # Extract or default core model parameters
        self.risk_free_rate = model_params.get("risk_free_rate", 0.0)
        self.q = model_params.get("dividend_yield", 0.0)

        directory = model_params['directory']
        low_strike = model_params['low_strike']
        high_strike = model_params['high_strike']
        low_maturity = model_params['low_maturity']
        high_maturity = model_params['high_maturity']
        N = model_params.get('N', 20)

        # Build the local volatility interpolator
        self.interp_local_vol, _, _, _ = build_local_vol_surface_from_implied(
            directory=directory,
            low_strike=low_strike,
            high_strike=high_strike,
            low_maturity=low_maturity,
            high_maturity=high_maturity,
            N=N
        )
    
    def _drift(self, price, t):
        return (self.risk_free_rate - self.q) * price

    def _diffusion(self, price, t):
        sigma_local = self.local_volatility(price, t)  # function of price and time
        return sigma_local * price

    def _diffusion_prime(self, price, t):
        sigma_local = self.local_volatility(price, t)
        d_sigma_d_price = self.local_vol_derivative(price, t)  # derivative of sigma_local w.r.t price
        return sigma_local + price * d_sigma_d_price
    
    def local_volatility(self, price, t):
        point = np.array([[t, price]])
        return self.interp_local_vol(point)[0]
    
    def local_vol_derivative(self, price, t, h=1e-4):
        return (self.local_volatility(price + h, t) - self.local_volatility(price - h, t)) / (2 * h)

#finite differences 
#git merge origin/
# main
#git kraken 