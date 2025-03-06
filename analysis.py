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

def plot_trajectory(directory, figsize=(14,10)):
    """
    Plot state trajectories from simulation data.

    Parameters
    ---
    directory : str
        Path to directory containing simulation data.
    figsize : tuple
        Size of figure to be plotted.
    """
    # Identify file paths
    t_file_path = os.path.join(directory, "t.npy")
    samples_file_path = os.path.join(directory, "samples.npy")
    # Load files
    t = np.load(t_file_path)
    samples = np.load(samples_file_path, allow_pickle=True).item()
    dim = len(samples)

    # Create figure and plot price and volatility on separate axes
    fig, ax = plt.subplots(dim, 1, figsize=figsize)
    ax = np.atleast_1d(ax)
    # Plot samples 
    for cpt in range(dim): # Loop over dimensions of model
        cpt_var = list(samples.keys())[cpt]
        data = samples[cpt_var]
        for i in range(data.shape[0]): # Loop over number of paths
            ax[cpt].plot(t, data[i, :], alpha=0.5)
            ax[cpt].set_xlabel('Time')
            ax[cpt].set_ylabel(cpt_var)
            ax[cpt].grid(True, alpha=0.5)
    plt.tight_layout()

    # Save plot to output file
    output_file = os.path.join(get_directory(t_file_path), "trajectory_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"Trajectory plot saved to {output_file}")

    plt.close()

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
    samples_file_path = os.path.join(directory, "samples.npy")
    params_file_path = os.path.join(directory, "params.json")
    # Load arrays
    t = np.load(t_file_path)
    samples = np.load(samples_file_path, allow_pickle=True).item()
    # Load parameter json
    with open(params_file_path, "r") as f:
        params = json.load(f)

    S = samples['S']
    # Protect against maturity outside simulated time horizon
    if maturity > params['final_time']:
        raise ValueError(f'Maturity must be between 0 and simulated final time. \n Provided: maturity = {maturity}, T = {params["T"]}')
    # Compute option price
    maturity_idx = np.searchsorted(t, maturity)
    N = S.shape[0]
    C_values = np.zeros(N)
    for i in range(N):
        C_values[i] = np.exp(-params['r'] * maturity) * max(S[i,maturity_idx] - K, 0)

    print(f'Option price: {np.mean(C_values):.2f}')
