import configparser
from utils.build_utils import parse_value, parse_possible_list, create_directory, load_class, list_files_excluding
from utils.sim_utils import print_section
import analysis as anl 
import os 
from models.black_scholes import BlackScholes
from models.heston import Heston
from models.cox_ingersoll_ross import CoxIngersollRoss
from models.ornstein_uhlenbeck import OrnsteinUhlenbeck
from simulators.euler_simulator import EulerSimulator
from simulators.milstein_simulator import MilsteinSimulator

def main(config_file):
    """
    Main function runs workflow based on an INI configuration file.

    Parameters
    ---
    config_file : str
        Path to the INI config file containing simulation parameters and options.
    """
    # Load configuration
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found. Available: {os.listdir('config_files')}")
    config = configparser.ConfigParser()
    config.read(config_file)
    # Run 
    model_name = config.get("run", "model_name")
    simulator_name = config.get("run", "simulator_name")
    # Model parameters 
    model_params = {key: parse_value(config.get("model_params", key)) 
                    for key in config.options("model_params")}
    # Simulation parameters
    simulator_params = {key : parse_value(config.get("simulation", key))
                        for key in config.options("simulation")}
    # Output parameters
    do_timestamp = config.getboolean("output", "do_timestamp")
    # Workflow controls
    run_simulation = config.getboolean("workflow", "run_simulation")
    run_analysis = config.getboolean("workflow", "run_analysis")
    # Option pricing parameters
    run_option_pricing = config.getboolean("option", "run_option_pricing")
    strike_price = config.getfloat("option", "strike_price")
    maturity = config.getfloat("option", "maturity")
    
    # Create output directory
    directory = create_directory(model_name=model_name, simulator_name=simulator_name, simulator_params=simulator_params, do_timestamp=do_timestamp)
    
    # Run simulation processes
    if run_simulation:
        model_class = globals().get(model_name)
        if model_class is None:
            raise ValueError(f"Model class '{model_name}' not found. Available: {list_files_excluding('models', 'model.py')}")
        model = model_class(model_params=model_params) # Instantiate model dynamically
        print_section(f"Initiating {simulator_name}  simulation of {model_name} model with {simulator_params['number_of_paths']} paths final_time={simulator_params['final_time']} and discretisation parameter n={simulator_params['discretisation_parameter']}...")

        simulator_class = globals().get(simulator_name)
        if simulator_class is None:
                raise ValueError(f'Simulator class {simulator_name} not found. Available: {list_files_excluding("simulators", "simulator.py")}')
        simulator = simulator_class(model=model, simulator_params=simulator_params) 
        
        simulator.sim(directory=directory) 
    
    if run_analysis:
        print_section(f'Analysing simulation results from {directory}...')
        anl.plot_trajectory(directory)
    
    if run_option_pricing:
        print_section(f'Pricing option with strike price K={strike_price} and maturity T_M={maturity}...')
        anl.price_option(directory, strike_price, maturity)
        anl.implied_volatility(directory)
    
    print_section("Workflow Complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run model workflow based on configuration file.")
    parser.add_argument('--config', type=str, default='config_files/black_scholes.ini', help="Path to the INI configuration file.")
    args = parser.parse_args()
    main(args.config)
