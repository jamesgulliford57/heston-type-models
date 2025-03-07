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
    plt.style.use('ggplot')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'figure.titlesize': 20,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        'legend.fontsize': 14,
    })
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

    cmap = plt.get_cmap('tab10')
    # Plot samples 
    for cpt in range(dim): # Loop over dimensions of model
        cpt_var = list(samples.keys())[cpt]
        data = samples[cpt_var]
        for path in range(data.shape[0]): # Loop over number of paths
            color = cmap(path % 10)
            ax[cpt].plot(t, data[path, :], lw=1.5, color=color, alpha=0.5)
        ax[cpt].set_ylabel(cpt_var.capitalize(), fontsize=18)
        ax[cpt].tick_params(axis='both', which='major', labelsize=14)
        ax[cpt].grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    
    ax[-1].set_xlabel('Time', fontsize=16)
    fig.suptitle('Simulated Trajectories', y=0.92, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot to output file
    output_file = os.path.join(get_directory(t_file_path), "trajectory_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"Trajectory plot saved to {output_file}")

    plt.close()

def price_option(directory, strike, maturity):
    """
    Compute average price for an option with strike price K and maturity T given simulation data.

    Parameters
    ---
    directory : str
        Path to directory containing simulation data.
    strike : float
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

    price = samples['price']
    risk_free_rate = params['risk_free_rate']
    # Protect against maturity outside simulated time horizon
    if maturity > params['final_time']:
        raise ValueError(f'Maturity must be between 0 and simulated final time. \n Provided: maturity = {maturity}, T = {params["T"]}')
    # Compute option price
    maturity_idx = np.searchsorted(t, maturity)
    num_paths = price.shape[0]
    C_values = np.zeros(num_paths)
    for i in range(num_paths):
        C_values[i] = np.exp(-risk_free_rate * maturity) * max(price[i,maturity_idx] - strike, 0)

    print(f'Option price: {np.mean(C_values):.2f}')
