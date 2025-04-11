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
    time_values_file_path = os.path.join(directory, "time_values.npy")
    samples_file_path = os.path.join(directory, "samples.npy")
    # Load files
    time_values = np.load(time_values_file_path)
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
            ax[cpt].plot(time_values, data[path, :], lw=1.5, color=color, alpha=0.5)
        ax[cpt].set_ylabel(cpt_var.capitalize(), fontsize=18)
        ax[cpt].tick_params(axis='both', which='major', labelsize=14)
        ax[cpt].grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    
    ax[-1].set_xlabel('Time', fontsize=16)
    fig.suptitle('Simulated Trajectories', y=0.92, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot to output file
    output_file = os.path.join(get_directory(time_values_file_path), "trajectory_plot.png")
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
    time_values_file_path = os.path.join(directory, "time_values.npy")
    samples_file_path = os.path.join(directory, "samples.npy")
    output_file_path = os.path.join(directory, "output.json")
    # Load arrays
    time_values = np.load(time_values_file_path)
    samples = np.load(samples_file_path, allow_pickle=True).item()
    # Load parameter json
    with open(output_file_path, "r") as f:
        output = json.load(f)

    price = samples['price']
    risk_free_rate = output['model_params']['risk_free_rate']
    # Protect against maturity outside simulated time horizon
    if maturity > output['final_time']:
        raise ValueError(f'Maturity must be between 0 and simulated final time. \n Provided: maturity = {maturity}, T = {output["T"]}')
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
    # Save option price to output file
    output['option'] = {
        'strike': strike,
        'maturity': maturity,
        'call_price': call_price,
        'put_price': put_price 
    }
    with open(output_file_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f'Call option price: {call_price:.2f}')
    print(f'Put option price: {put_price:.2f}')

def price_call_black_scholes(stock_price, strike, maturity, risk_free_rate, sigma):
    """
    Calculate the Black-Scholes call option price. Test numerically obtained 
    option price against analytical solution or use to determine implied volatility.

    Parameters
    ---
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

    Returns
    ---
    float
        Call option price.
    """
    from scipy.stats import norm
    d1 = (np.log(stock_price / strike) + (risk_free_rate + 0.5 * sigma ** 2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return stock_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate * maturity) * norm.cdf(d2)

def implied_volatility(directory):
    """
    Compute implied volatility from option price and model parameters.

    Parameters
    ---
    directory : str
        Path to directory containing simulation data.
    """
    from scipy.optimize import brentq
    # Identify file paths
    output_file_path = os.path.join(directory, "output.json")
    # Load parameter json
    with open(output_file_path, "r") as f:
        output = json.load(f)
    
    # Extract parameters from output
    risk_free_rate = output['model_params']['risk_free_rate']
    strike = output['option']['strike']
    maturity = output['option']['maturity']
    call_price = output['option']['call_price']
    stock_price = output['initial_value'][0]
    
    # Compute implied volatility using Black-Scholes formula
    objective_function = lambda sigma: price_call_black_scholes(stock_price=stock_price, strike=strike, maturity=maturity, risk_free_rate=risk_free_rate, sigma=sigma) - call_price
    implied_vol = brentq(objective_function, 1e-6, 1.0)
    print(f'Implied volatility: {implied_vol:.2f}')

    with open(output_file_path, 'w') as f:
        output['implied_volatility'] = implied_vol
        json.dump(output, f, indent=4)

    return implied_vol