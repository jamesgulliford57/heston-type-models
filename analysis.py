import os
import numpy as np
import matplotlib.pyplot as plt
import json

def get_directory(file_path):
    """
    Obtain directory provided file is stored at.

    Parameters
    ---
    file_path : str
        Path to file
    """
    return os.path.dirname(file_path)

def plot_trajectory(directory, figsize=(10,10)):
    """
    Plot stock price and implied volatility trajectories from simulation data.
    (Specific to Heston model (S,V) at present, needs to be generalised)

    Parameters
    ---
    directory : str
        Path to directory containing simulation data.
    figsize : tuple
        Size of figure to be plotted.
    """
    # Identify file paths
    t_file_path = os.path.join(directory, "t.npy")
    S_file_path = os.path.join(directory, "S.npy")
    V_file_path = os.path.join(directory, "V.npy")
    # Load files
    t = np.load(t_file_path)
    S = np.load(S_file_path)
    V = np.load(V_file_path)

    # Create figure and plot price and volatility on separate axes
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=figsize)
    # Plot samples 
    for i in range(S.shape[0]):
        ax1.plot(t, S[i], alpha=0.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.set_title("Stock price")
    ax1.grid(True, alpha=0.5)
    for i in range(V.shape[0]): 
        ax2.plot(t, V[i], alpha=0.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volatility')
    ax2.set_title("Implied volatility")
    ax2.grid(True, alpha=0.5)
    plt.tight_layout()

    # Save plot to output file
    output_file = os.path.join(get_directory(t_file_path), "trajectory_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"Trajectory plot saved to {output_file}")

def price_option(directory, K, maturity):
    """
    Compute average price for an option with strike price K and maturity T given simulation data.

    Parameters
    ---
    directory : str
        Path to directory containing simulation data.
    K : float
        Option strike price.
    maturity : float
        Option maturity.
    """
    # Identify file paths
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
    # Compute option price
    maturity_idx = np.searchsorted(t, maturity)
    N = S.shape[0]
    C_values = np.zeros(N)
    for i in range(N):
        C_values[i] = np.exp(-params['r'] * maturity) * max(S[i,maturity_idx] - K, 0)

    print(f'Option price: {np.mean(C_values):.2f}')
