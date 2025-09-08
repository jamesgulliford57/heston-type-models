import os
import sys
import json
import numpy as np


def get_strong_error(directory, time_value, exact_solution_function=None):
    """
    Compute strong error for set of simulation samples by comparing against exact solution.

    Parameters
    ----------
    directory : str
        Path to directory containing simulation data.
    time_value : float
        Point in time to evaluate the strong error.
    exact_solution_function : callable, optional
        Function that computes exact solution. If None, uses Black-Scholes exact solution.
    """
    params_file_path = os.path.join(directory, 'params.json')
    with open(params_file_path, 'r') as f:
        params = json.load(f)
    
    model_name = params['model_name']
    samples_file_path = os.path.join(directory, "samples.npy")
    samples = np.load(samples_file_path, allow_pickle=True).item()
    
    time_values = samples["time"]
    time_value_idx = np.searchsorted(time_values, time_value)
    
    # Extract price samples (assuming shape: [num_paths, num_time_steps])
    price_samples = samples['price']
    
    # Get exact solution
    if exact_solution_function is not None:
        exact_values = exact_solution_function(params, time_value)
    else:
        # Default: Black-Scholes exact solution
        if "BlackScholes" not in model_name:
            raise ValueError(f"Default exact solution only available for BlackScholes model. Provided: {model_name}. Use exact_solution_function parameter.")
        exact_values = _black_scholes_exact_solution(params, time_value, price_samples.shape[0])
    
    # Extract samples at the specific time point
    sample_at_time_value = price_samples[:, time_value_idx]
    
    # Calculate strong error: mean absolute difference
    strong_error = np.mean(np.abs(sample_at_time_value - exact_values))
    
    print(f'Strong Error at time {time_value}: {strong_error:.6f}')
    
    return strong_error


def _black_scholes_exact_solution(params, time_value, num_paths):
    """
    Compute exact solution for Black-Scholes model.
    For strong error, we need path-wise exact solution using the same Brownian paths.
    This requires storing Brownian paths during simulation.
    """
    # Check if Brownian paths are stored
    samples_file_path = os.path.join(os.path.dirname(params.json), "samples.npy")
    samples = np.load(samples_file_path, allow_pickle=True).item()
    
    if "brownian" not in samples:
        raise ValueError("Brownian paths not found in samples. Strong error calculation requires path-wise comparison.")
    
    brownian_paths = samples['brownian']
    time_values = samples["time"]
    time_value_idx = np.searchsorted(time_values, time_value)
    
    # Extract Brownian motion at the specific time
    brownian_at_time = brownian_paths[:, time_value_idx]
    
    initial_value = params["initial_value"][0]
    risk_free_rate = params["model_params"]["risk_free_rate"]
    volatility = params["model_params"]["volatility"]
    dividend_yield = params["model_params"].get("q", 0.0)
    
    # Exact solution for GBM: X_t = X_0 * exp((r - q - σ²/2)t + σW_t)
    drift = (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_value
    diffusion = volatility * brownian_at_time
    
    exact_values = initial_value * np.exp(drift + diffusion)
    
    return exact_values


def get_strong_error_convergence(directory, time_value, n_values, exact_solution_function=None):
    """
    Compute strong error for multiple discretization levels to study convergence.
    
    Parameters
    ----------
    directory : str
        Base directory containing subdirectories for different n values.
    time_value : float
        Point in time to evaluate the strong error.
    n_values : list of int
        List of discretization parameters to test.
    exact_solution_function : callable, optional
        Function that computes exact solution.
    """
    strong_errors = []
    
    for n in n_values:
        subdir = os.path.join(directory, f"n_{n}")
        if not os.path.exists(subdir):
            print(f"Warning: Directory {subdir} not found. Skipping n={n}.")
            continue
        
        try:
            error = get_strong_error(subdir, time_value, exact_solution_function)
            strong_errors.append((n, error))
        except Exception as e:
            print(f"Error processing n={n}: {e}")
    
    return np.array(strong_errors)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Usage: python get_strong_error.py <directory> <time_value> [<exact_solution_function>]")
    
    directory = sys.argv[1]
    time_value = float(sys.argv[2])
    
    if len(sys.argv) == 4:
        # This would require a more sophisticated way to pass functions
        # For simplicity, we assume built-in exact solutions for now
        exact_func_name = sys.argv[3]
        if exact_func_name == "BlackScholes":
            exact_solution_function = None  # Use default
        else:
            raise ValueError(f"Unknown exact solution function: {exact_func_name}")
        get_strong_error(directory=directory, time_value=time_value, 
                        exact_solution_function=exact_solution_function)
    else:
        get_strong_error(directory=directory, time_value=time_value)