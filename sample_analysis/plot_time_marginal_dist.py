import os
import sys
import json
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import get_color_map


def plot_time_marginal_dist(directories, marginal_time, figsize=(12, 8)):
    """
    Plot time marginal distribution of process from simulation data. Provides option to compare against
    analytical distribution to test for correct behaviour.

    Parameters
    ----------
    directory : str
        Path to directory containing simulation data.
    time : float
        Time at which to evaluate marginal distribution.
    figsize : tuple
        Size of figure to be plotted.
    """
    num_directories = len(directories)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = get_color_map(num_directories)[0]

    for index, directory in enumerate(directories):
        params_file_path = os.path.join(directory, 'params.json')
        with open(params_file_path, 'r') as f:
            params = json.load(f)
        simulator_name = params['simulator_name']
        model_name = params['model_name']
        final_time = params['final_time']

        samples_path = os.path.join(directory, 'samples.npy')
        samples = np.load(samples_path, allow_pickle=True).item()
        time_values = samples["time"]
        del samples["time"]
        price = samples["price"]

        if marginal_time > time_values[-1]:
            raise ValueError('Time to evaluate marginal distribution must be less than or equal to final_time. '
                             f'Final_time {final_time}')
        time_index = np.searchsorted(time_values, marginal_time)
        marginal_prices = price[:, time_index]

        ax.hist(marginal_prices, bins=100, density=True, color=colors[index], alpha=0.5, edgecolor='black',
                label=f'Empirical\nModel: {model_name}\n{simulator_name}\n{len(marginal_prices)} paths')

        if model_name == 'BlackScholes':
            initial_value = params['initial_value']
            q = params['model_params']['q']
            sigma = params['model_params']['sigma']
            risk_free_rate = params['model_params']['risk_free_rate']
            mean = np.log(initial_value) + ((risk_free_rate - q) - 0.5 * sigma ** 2) * marginal_time
            variance = sigma ** 2 * marginal_time
            lognorm_dist = lognorm(s=variance**0.5, scale=np.exp(mean))
            x = np.linspace(min(marginal_prices), max(marginal_prices), 10000)
            lognorm_pdf = lognorm_dist.pdf(x)
            ax.plot(x, lognorm_pdf, color=colors[index], linestyle='--', label=f'Analytical\nModel: {model_name}')


    ax.set_xlabel('Price')
    ax.set_ylabel('Marginal PDF')
    ax.set_title(f'Marginal distribution: Time={marginal_time:.2g}')
    ax.grid(True, alpha=0.6)
    ax.legend()

    for directory in directories:
        output_file = os.path.join(directory, "time_marginal_dist.png")
        plt.savefig(output_file, dpi=400)
        print(f"{output_file} saved.")

    plt.close()

    return fig, ax

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Usage: python plot_time_marginal_dist.py <directory1> [<directory2> ...] <marginal_time>")
    *directories, marginal_time = sys.argv[1:]
    marginal_time = float(marginal_time)
    plot_time_marginal_dist(directories=directories, marginal_time=marginal_time)
