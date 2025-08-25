import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(directory, figsize=(14, 10)):
    """
    Plot state trajectories from simulation data.

    Parameters
    ----------
    directory : str
        Path to directory containing simulation data.
    figsize : tuple
        Size of figure to be plotted.
    """
    samples_file_path = os.path.join(directory, "samples.npy")
    samples = np.load(samples_file_path, allow_pickle=True).item()
    time_values = samples["time"]
    del samples["time"]
    dim = len(samples)
    # Create figure
    fig, ax = plt.subplots(dim, 1, figsize=figsize)
    ax = np.atleast_1d(ax)
    cmap = plt.get_cmap('tab10')
    # Plot samples
    for component_index, state_component in enumerate(samples.keys()):
        sample = samples[state_component]
        for path_index, path in enumerate(sample):
            color = cmap(path_index % 10)
            ax[component_index].plot(time_values, path, lw=1.5, color=color, alpha=0.5)
        ax[component_index].set_ylabel(state_component.capitalize(), fontsize=18)
        ax[component_index].tick_params(axis='both', which='major', labelsize=14)
        ax[component_index].grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    ax[-1].set_xlabel('Time', fontsize=16)
    fig.suptitle('Simulated Trajectories', y=0.94, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    # Save plot to output file
    output_file = os.path.join(directory, "trajectory_plot.png")
    plt.savefig(output_file, dpi=400)
    print(f"{output_file} saved.")

    plt.close()

    return fig, ax


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python plot_trajectory.py <directory>")
    plot_trajectory(sys.argv[1])
