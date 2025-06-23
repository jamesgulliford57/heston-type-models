import os
import sys
import json
import numpy as np


def price_option(directory, strike, maturity):
    """
    Compute average price for an option with strike price K and maturity T given simulated paths.

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
    output_file_path = os.path.join(directory, "output.json")
    with open(output_file_path, "r") as f:
        output = json.load(f)
    price = samples['price']
    risk_free_rate = output['model_params']['risk_free_rate']
    strike, maturity = float(strike), float(maturity)
    # Protect against maturity outside simulated time horizon
    if maturity > output['final_time']:
        raise ValueError(f'Maturity must be between 0 and simulated final time. \n'
                         f'Provided: maturity = {maturity}, T = {output["T"]}')
    # Compute option price
    maturity_idx = np.searchsorted(time_values, maturity)
    num_paths = price.shape[0]
    C_values = np.zeros(num_paths)
    P_values = np.zeros(num_paths)
    for i in range(num_paths):
        C_values[i] = np.exp(-risk_free_rate * maturity) * max(price[i,maturity_idx] - strike, 0)
        P_values[i] = np.exp(-risk_free_rate * maturity) * max(strike - price[i,maturity_idx], 0)
    call_price = np.mean(C_values)
    put_price = np.mean(P_values)

    print(f'Call option price: {call_price:.2f}')
    print(f'Put option price: {put_price:.2f}')

    return call_price, put_price

if __name__ == "__main__":
    price_option(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
