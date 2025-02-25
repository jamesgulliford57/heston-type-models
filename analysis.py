import os
import numpy as np
import matplotlib.pyplot as plt
import json

def get_directory(file_path):
    return os.path.dirname(file_path)

def plot_trajectory(directory):
    # Identify file paths
    t_file_path = os.path.join(directory, "t.npy")
    S_file_path = os.path.join(directory, "S.npy")
    V_file_path = os.path.join(directory, "V.npy")
    # Load files
    t = np.load(t_file_path)
    S = np.load(S_file_path)
    V = np.load(V_file_path)
    # Create figure and plot price and volatility on separate axex
    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    for i in range(S.shape[0]):
        plt.plot(t, S[i], alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title("Stock price")
    plt.grid(True, alpha=0.5)
    plt.subplot(212)
    for j in range(V.shape[0]):
        plt.plot(t, V[j], alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.title("Implied volatility")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()

    output_file = os.path.join(get_directory(t_file_path), "trajectory_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"Trajectory plot saved to {output_file}")

def price_option(directory, K, maturity):
    """
    Compute average price for an option with strike price K and maturity T using 
    """
    t_file_path = os.path.join(directory, "t.npy")
    S_file_path = os.path.join(directory, "S.npy")
    params_file_path = os.path.join(directory, "params.json")
    # Load arrays
    S = np.load(S_file_path)
    t = np.load(t_file_path)
    # Load parameter json
    with open(params_file_path, "r") as f:
        params = json.load(f)
    # Protect against maturity outside simulated time horizon
    if maturity > params['T']:
        raise ValueError(f'Maturity must be between 0 and simulated horizon T. \n Provided: maturity = {maturity}, T = {params["T"]}')
    maturity_idx = np.searchsorted(t, maturity)
    N = S.shape[0]
    C_values = np.zeros(N)
    for i in range(N):
        C_values[i] = np.exp(-params['r'] * maturity) * max(S[i,maturity_idx] - K, 0)

    print(f'Option price: {np.mean(C_values):.2f}')

# Specify the directory containing the files
#directory = "data/euler_T=5_n=10000_init=[1.   0.16]_N=100/15012025_213126"
#print(f"Data loaded from directory: {directory}")

#plot_trajectory(directory)
#price_option(directory, 100, 5)

"""
i) Finish config file - add way to choose which plots to plot
ii) Option pricing and volatility smile
iii) Calculation of MSE and cost, comparison of different schemes
iv) Plot line on stock price graph to represent price of option that was priced using price_option
"""