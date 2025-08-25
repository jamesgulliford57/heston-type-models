import sys
import numpy as np
from scipy.stats import norm


def black_scholes_vega(stock_price, strike, maturity, risk_free_rate, sigma, q=0.0):
    """
    Calculate vega in accordance with Black-Scholes model.

    Parameters
    ----------
    stock_price : float
        Current stock price.
    strike : float
        Option strike price.
    maturity : float
        Time to maturity.
    risk_free_rate : float
        Risk-free rate.
    sigma : float
        Volatility of the underlying asset.
    q : float
        Dividend yield.
    """
    d1 = (np.log(stock_price / strike) + (risk_free_rate - q + 0.5 * sigma ** 2) * maturity) / (sigma * np.sqrt(maturity))
    return stock_price * norm.pdf(d1) * np.sqrt(maturity)

