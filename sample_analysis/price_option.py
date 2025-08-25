import os
import sys
import json
import numpy as np


def price_option(directory, strike, maturity):
    """
    Compute average price for European call option with strike price K and maturity T for given simulation samples.

    Parameters
    ----------
    directory : str
        Path to directory containing simulation data.
    strike : float
        Option strike price.
    maturity : float
        Option maturity.
    """
    samples_file_path = os.path.join(directory, "samples.npy")
    samples = np.load(samples_file_path, allow_pickle=True).item()
    time_values = samples["time"]
    del samples["time"]
    params_file_path = os.path.join(directory, "params.json")
    with open(params_file_path, "r") as f:
        params = json.load(f)
    if maturity > params['final_time'] or maturity < 0:
        raise ValueError(f'Maturity must be between 0 and simulated final time. \n'
                         f'Provided: maturity={maturity}, final_time={params["final_time"]}')
    if strike < 0:
        raise ValueError(f'Strike price must not be negative. Provided: {strike}')
    price = samples['price']
    risk_free_rate = params['model_params']['risk_free_rate']
    # Compute option price
    maturity_idx = np.searchsorted(time_values, maturity)
    prices_at_maturity = price[:, maturity_idx]
    payoffs = np.maximum(prices_at_maturity - strike, 0)
    discount_factor = np.exp(-risk_free_rate * maturity)
    call_price = discount_factor * np.mean(payoffs)

    n = np.shape(price)[0]
    call_price_error = discount_factor * np.std(payoffs) / np.sqrt(n)

    print(f'European call (K={strike:.2f}, T={maturity:.2g}) price: {call_price:.2f} +- {call_price_error:.2f}')

    return call_price, call_price_error


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Usage: python price_option.py <directory> <strike> <maturity>")
    directory = sys.argv[1]
    strike = float(sys.argv[2])
    maturity = float(sys.argv[3])
    price_option(directory=directory, strike=strike, maturity=maturity)
