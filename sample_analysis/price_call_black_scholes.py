import sys
import numpy as np
from scipy.stats import norm


def price_call_black_scholes(stock_price, strike, maturity, risk_free_rate, sigma):
    """
    Calculate the Black-Scholes call option price. Test numerically obtained
    option price against analytical solution or use to determine implied volatility.

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
    """
    d1 = (np.log(stock_price / strike) + (risk_free_rate + 0.5 * sigma ** 2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    call_price = stock_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * maturity) * norm.cdf(d2)
    return call_price

if __name__ == "__main__":
    stock_price = float(sys.argv[1])
    strike = float(sys.argv[2])
    maturity = float(sys.argv[3])
    risk_free_rate = float(sys.argv[4])
    sigma = float(sys.argv[5])
    price_call_black_scholes(stock_price=stock_price, strike=strike, maturity=maturity,
                             risk_free_rate=risk_free_rate, sigma=sigma)
