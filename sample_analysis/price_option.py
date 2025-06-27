import os
import sys
import json
import numpy as np


def price_option(directory, strike, maturity):
    """
    Compute average price for European call option with strike price K and maturity T given simulated paths.

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
    price = samples['price']
    risk_free_rate = params['model_params']['risk_free_rate']
    # Protect against maturity outside simulated time horizon
    if maturity > params['final_time']:
        raise ValueError(f'Maturity must be between 0 and simulated final time. \n'
                         f'Provided: maturity={maturity}, T={params["T"]}')
    # Compute option price
    maturity_idx = np.searchsorted(time_values, maturity)
    num_paths = price.shape[0]
    C_values = np.zeros(num_paths)
    for path_index in range(num_paths):
        C_values[path_index] = np.exp(-risk_free_rate * maturity) * max(price[path_index, maturity_idx] - strike, 0)
    call_price = np.mean(C_values)

    print(f'European call (K={strike:.2f}, T={maturity:.2g}) price: {call_price:.2f}')

    return call_price

if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Usage: python price_option.py <directory> <strike> <maturity>")
    directory = sys.argv[1]
    strike = float(sys.argv[2])
    maturity = float(sys.argv[3])
    price_option(directory=directory, strike=strike, maturity=maturity)
