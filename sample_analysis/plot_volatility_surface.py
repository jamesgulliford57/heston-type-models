import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from implied_volatility import implied_volatility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import get_color_map


def plot_volatility_surface(directories, low_strike, high_strike, low_maturity, high_maturity, figsize=(12, 8)):
    """
    Plot implied volatility as a function of strike price and maturity from simulation data.

    Parameters
    ----------
    directory : str
        Path to directory containing simulation data.
    low_strike : float
        Lowest strike to loop over.
    high_strike : float
        Highest strike to loop over.
    low_maturity : float
        Lowest maturity of output.
    high_maturity : float
        Highest maturity of output.
    figsize : tuple
        Size of figure to be plotted.
    """
    num_directories = len(directories)
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"projection": "3d"})
    cmap = get_color_map(num_directories)[1]

    n_strikes, n_maturities = 20, 20
    strikes = np.linspace(low_strike, high_strike, n_strikes)
    maturities = np.linspace(low_maturity, high_maturity, n_maturities)
    for directory in directories:
        params_file_path = os.path.join(directory, 'params.json')
        with open(params_file_path, 'r') as f:
            params = json.load(f)
        simulator_name = params['simulator_name']
        model_name = params['model_name']
        implied_volatilities, implied_volatility_errors = (np.zeros((n_strikes, n_maturities)),
                                                           np.zeros((n_strikes, n_maturities)))
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                implied_volatilities[i, j], implied_volatility_errors[i, j] = implied_volatility(directory=directory,
                                                                                                 strike=strike,
                                                                                                 maturity=maturity)
        strikes, maturities = np.meshgrid(strikes, maturities)
        ax.plot_surface(strikes, maturities, implied_volatilities.T, cmap=cmap, alpha=0.7, antialiased=True)

    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'European Call Option Volatility Surface: {model_name} {simulator_name}')

    for directory in directories:
        output_file = os.path.join(directory, "volatility_surface.png")
        plt.savefig(output_file, dpi=400)
        print(f"{output_file} saved.")

    plt.show()
    plt.close()

    return fig, ax


if __name__ == "__main__":
    if len(sys.argv) < 6:
        raise ValueError("Usage: python plot_volatility_surface.py <directory1> [<directory2> ...] <low_strike> "
                         "<high_strike> <low_maturity> <high_maturity>")
    *directories, low_strike, high_strike, low_maturity, high_maturity = sys.argv[1:]
    low_strike = float(low_strike)
    high_strike = float(high_strike)
    low_maturity = float(low_maturity)
    high_maturity = float(high_maturity)
    plot_volatility_surface(directories=directories, low_strike=low_strike, high_strike=high_strike,
                            low_maturity=low_maturity, high_maturity=high_maturity)
