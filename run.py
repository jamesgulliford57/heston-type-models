import os
import sys
import configparser
from utils.build_utils import parse_value, list_files_excluding
from models.heston import Heston
from models.black_scholes import BlackScholes
from models.cox_ingersoll_ross import CoxIngersollRoss
from models.ornstein_uhlenbeck import OrnsteinUhlenbeck
from simulators.euler_simulator import EulerSimulator
from simulators.milstein_simulator import MilsteinSimulator


def main(config_path):
    """
    Run simulation. Parameters and methods set by config_path.

    Parameters
    ----------
    config_path : str
        Path to config file.
    """
    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found. "
                                f"Available: {os.listdir('config_paths')}")
    config = configparser.ConfigParser()
    config.read(config_path)
    # Load parameters
    model_name = config.get("run", "model_name")
    simulator_name = config.get("run", "simulator_name")
    model_params = {key: parse_value(config.get("model_params", key)) for key in config.options("model_params")}
    simulator_params = {key: parse_value(config.get("simulation", key)) for key in config.options("simulation")}
    directory = config.get("output", "output_directory")
    os.makedirs(directory, exist_ok=True)
    # Instantiate model
    model_class = globals().get(model_name)
    if model_class is None:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list_files_excluding('models', 'model.py')}")
    model = model_class(model_params=model_params)
    # Instantiate simulator
    simulator_class = globals().get(simulator_name)
    if simulator_class is None:
        raise ValueError(f"Simulator class {simulator_name} not found."
                         f"Available: {list_files_excluding('simulators', 'simulator.py')}")
    simulator = simulator_class(model=model, simulator_params=simulator_params)
    # Perform simulation
    print(f"Initiating {simulator_name} simulation of {model_name} model with "
          f"{simulator_params['number_of_paths']} paths, final time={simulator_params['final_time']} and "
          f"discretisation parameter n={simulator_params['discretisation_parameter']}.")
    simulator.sim(directory=directory)

    print(f"{simulator_name} simulation of {model_name} model complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python run.py <directory> <config_path>")
    config_path = sys.argv[1]
    main(config_path=config_path)
