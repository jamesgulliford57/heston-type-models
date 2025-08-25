import os
import sys
import json
import numpy as np


def get_weak_error(directory, time_value, test_function=lambda x: x):
    """
    Compute weak error for set of simulation samples.

    Parameters
    ----------
    directory : str
        Path to directory containing simulation data.
    time_value : float
        Point in time to evaluate the weak error.
    test_function : callable
        Function to evaluate weak error.
    """
    params_file_path = os.path.join(directory, 'params.json')
    with open(params_file_path, 'r') as f:
        params = json.load(f)
    model_name = params['model_name']
    if "BlackScholes" not in model_name:
        raise ValueError(f"Weak error can only be computed for BlackScholes model simulations. Provided: {model_name}")
    samples_file_path = os.path.join(directory, "samples.npy")
    samples = np.load(samples_file_path, allow_pickle=True).item()
    time_values = samples["time"]
    time_value_idx = np.searchsorted(time_values, time_value)
    del samples["time"]
    samples = samples['price']
    initial_value = params["initial_value"][0]
    drift = params["model_params"]["risk_free_rate"] - params["model_params"]["q"]

    expected_xt = initial_value * np.exp(drift * time_value)
    expected_test_function = test_function(expected_xt)

    sample_at_time_value = samples[:, time_value_idx]
    test_function_output = [test_function(x) for x in sample_at_time_value]
    agg_test_function_output = np.mean(test_function_output)

    weak_error = abs(agg_test_function_output - expected_test_function)

    print(f'Weak Error: {weak_error:.4f}')

    return weak_error


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Usage: python get_weak_error.py <directory> <time_value> <test_function>")
    directory = sys.argv[1]
    time_value = float(sys.argv[2])
    if len(sys.argv) == 4:
        test_function = sys.argv[3]
        get_weak_error(directory=directory, time_value=time_value, test_function=test_function)
    else:
        get_weak_error(directory=directory, time_value=time_value)
