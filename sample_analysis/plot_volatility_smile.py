import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from price_option import price_option
from implied_volatility import implied_volatility


def plot_volatility_smile(directory, low_strike, high_strike, maturity, figsize=(12, 8)):
    """
    Plot state trajectories from simulation data.

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
    strikes = np.linspace(low_strike, high_strike, 100)
    option_prices = [price_option(directory=directory, strike=strike, maturity=maturity)
                            for strike in strikes]
    implied_volatilities = [implied_volatility(directory=directory, strike=strike, maturity=maturity)
                            for strike in strikes]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(strikes, implied_volatilities, color='firebrick', label=f'Maturity: {maturity}')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied Volatility')
    ax.set_title('European Call Option Volatility Smile')
    ax.grid(True, alpha=0.6)
    ax.legend()

    output_file = os.path.join(directory, "volatility_smile.png")
    plt.savefig(output_file, dpi=400)
    print(f"{output_file} saved.")

    plt.close()

if __name__ == "__main__":
    directory = sys.argv[1]
    low_strike = float(sys.argv[2])
    high_strike = float(sys.argv[3])
    maturity = float(sys.argv[4])
    plot_volatility_smile(directory=directory, low_strike=low_strike, high_strike=high_strike, maturity=maturity)
