import os
import json
import numpy as np
from implied_volatility import implied_volatility
from scipy.interpolate import RegularGridInterpolator


def build_implied_vol_interpolator(directory, low_strike, high_strike, low_maturity, high_maturity, N=20):
    """
    Computes an implied volatility surface and returns a RegularGridInterpolator over it.

    Parameters
    ----------
    directory : str
        Path to directory containing simulation data.
    low_strike : float
        Lowest strike to loop over.
    high_strike : float
        Highest strike to loop over.
    low_maturity : float
        Lowest maturity to loop over.
    high_maturity : float
        Highest maturity to loop over.
    N : int
        Number of grid points in each dimension.

    Returns
    -------
    interpolator : RegularGridInterpolator
        Interpolator for implied volatility over (maturity, strike).
    strike_grid : np.ndarray
        Array of strike values.
    maturity_grid : np.ndarray
        Array of maturity values.
    surface : np.ndarray
        2D array of implied volatilities.
    """

    strike_grid = np.linspace(low_strike, high_strike, N)
    maturity_grid = np.linspace(low_maturity, high_maturity, N)

    surface = np.zeros((N, N))

    for i, strike in enumerate(strike_grid):
        for j, maturity in enumerate(maturity_grid):
            iv = implied_volatility(directory, strike, maturity)
            surface[i, j] = iv if iv is not None else np.nan

    # Transpose so surface[j, i] corresponds to surface[maturity, strike]
    surface = surface.T

    # Clean NaNs by simple forward fill (could also use interpolation later)
    if np.isnan(surface).any():
        print("Warning: Some implied vols were NaN. Filling with nearest neighbors.")
        mask = ~np.isnan(surface)
        surface = np.where(mask, surface, np.interp(np.flatnonzero(~mask), np.flatnonzero(mask), surface[mask]))

    interpolator = RegularGridInterpolator(
        (maturity_grid, strike_grid), surface, bounds_error=False, fill_value=None
    )

    return interpolator, strike_grid, maturity_grid, surface


def build_local_vol_surface_from_implied(interp_implied_vol, strike_grid, maturity_grid):
    T_grid, K_grid = np.meshgrid(maturity_grid, strike_grid, indexing='ij')
    local_vol_surface = np.zeros_like(T_grid)

    dK = (strike_grid[1] - strike_grid[0])
    dT = (maturity_grid[1] - maturity_grid[0])

    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            T = T_grid[i, j]
            K = K_grid[i, j]

            if T < dT or K < dK or K > strike_grid[-2] or T > maturity_grid[-2]:
                local_vol_surface[i, j] = np.nan
                continue

            sigma = interp_implied_vol([[T, K]])[0]

            sigma_T_plus = interp_implied_vol([[T + dT, K]])[0]
            sigma_T_minus = interp_implied_vol([[T - dT, K]])[0]
            d_sigma_dT = (sigma_T_plus - sigma_T_minus) / (2 * dT)

            sigma_K_plus = interp_implied_vol([[T, K + dK]])[0]
            sigma_K_minus = interp_implied_vol([[T, K - dK]])[0]
            d_sigma_dK = (sigma_K_plus - sigma_K_minus) / (2 * dK)

            d2_sigma_dK2 = (sigma_K_plus - 2 * sigma + sigma_K_minus) / (dK ** 2)

            numerator = sigma**2 + 2 * T * sigma * d_sigma_dT
            denominator = (1 - K * d_sigma_dK / sigma) ** 2 + K**2 * T * d2_sigma_dK2

            if denominator <= 0:
                local_vol_surface[i, j] = np.nan
            else:
                local_vol_surface[i, j] = np.sqrt(numerator / denominator)

    return local_vol_surface