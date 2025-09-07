import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from get_weak_error import get_weak_error


def plot_weak_error_vs_h(directories, time_value, test_function=lambda x: x):
    """
    Plot weak error as a function of discretisation interval.

    Parameters
    ----------
    directories : lst
        List of paths to directories containing simulation data.
    time_value : float
        Point in time to evaluate the weak error.
    test_function : callable
        Function to evaluate weak error.
    """
    h_values, weak_errors = [], []
    for directory in directories:
        params_file_path = os.path.join(directory, 'params.json')
        with open(params_file_path, 'r') as f:
            params = json.load(f)
        discretisation_parameter = params["discretisation_parameter"]
        final_time = params["final_time"]
        h = final_time / float(discretisation_parameter)
        h_values.append(h)
        weak_error = get_weak_error(directory=directory, time_value=time_value, test_function=test_function)
        weak_errors.append(weak_error)
    h_values.sort()
    fig, ax = plt.subplots()
    ax.plot(h_values, weak_errors, marker='8', color='firebrick')
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('h', fontsize=14)
    ax.set_ylabel('Weak Error', fontsize=14)
    ax.set_title('Weak Error vs Discretisation Interval')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Usage: python plot_weak_error_vs_h.py <directories> <time_value> <test_function>")
    *directories, time_value = sys.argv[1:]
    time_value = float(time_value)
    plot_weak_error_vs_h(directories=directories, time_value=time_value)

