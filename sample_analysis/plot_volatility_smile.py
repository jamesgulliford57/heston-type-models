import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from implied_volatility import implied_volatility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import get_color_map


def plot_volatility_smile(directories, low_strike, high_strike, maturity, figsize=(12, 8)):
    """
    Plot implied volatility as a function of strike price from simulation data.

    Parameters
    ----------
    directory : str
        Path to directory containing simulation data.
    low_strike : float
        Lowest strike to loop over.
    high_strike : float
        Highest strike to loop over.
    maturity : float
        Maturity of output.
    figsize : tuple
        Size of figure to be plotted.
    """
    num_directories = len(directories)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = get_color_map(num_directories)[0]

    strikes = np.linspace(low_strike, high_strike, 100)
    for index, directory in enumerate(directories):
        params_file_path = os.path.join(directory, 'params.json')
        with open(params_file_path, 'r') as f:
            params = json.load(f)
        simulator_name = params['simulator_name']
        model_name = params['model_name']

        implied_volatilities = [implied_volatility(directory=directory, strike=strike, maturity=maturity)
                                for strike in strikes]

        ax.plot(strikes, implied_volatilities, color=colors[index], label=f'Model: {model_name}\n{simulator_name}')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied Volatility')
    ax.set_title(f'European Call Option Volatility Smile: Maturity={maturity:.2g}')
    ax.grid(True, alpha=0.6)
    ax.legend()

    for directory in directories:
        output_file = os.path.join(directory, "volatility_smile.png")
        plt.savefig(output_file, dpi=400)
        print(f"{output_file} saved.")

    plt.close()

    return fig, ax

if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise ValueError("Usage: python plot_volatility_smile.py <directory1> [<directory2> ...] <low_strike> "
                         "<high_strike> <maturity>")
    *directories, low_strike, high_strike, maturity = sys.argv[1:]
    low_strike = float(low_strike)
    high_strike = float(high_strike)
    maturity = float(maturity)
    plot_volatility_smile(directories=directories, low_strike=low_strike, high_strike=high_strike, maturity=maturity)
