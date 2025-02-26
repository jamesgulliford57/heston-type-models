import os 
import json 
from datetime import datetime
from utils import print_section
from models.heston import HestonModel 
from analysis import plot_trajectory, price_option

def main(config_file):
    """
    Main function runs workflow based on config file 'config.json'.

    Parameters
    ---
    config_file : str
        Path to config file containing simulation parameters and options

    Workflow
    ---
        - Initialises selected model with provided parameters.
        - Creates directory for saving simulation results and parameters.
        - Runs simulation, analysis, and option pricing based on the config.
    """
    # Load configuration from JSON file 
    with open(config_file, 'r') as f:
        config = json.load(f)
    # Model parameters
    heston_params = config["heston_params"] 
    # Simulation parameters
    init_value = config["init_value"] 
    T = config["T"]
    n = config["n"]
    N = config["N"]
    scheme = config["scheme"]
    strike_price = config["strike_price"]
    maturity = config["maturity"]
    # Directories
    output_dir = config["output_dir"]
    do_timestamp = config["do_timestamp"]
    # Simulation controls
    run_simulation = config["run_simulation"]
    run_analysis = config["run_analysis"]
    run_option_pricing = config["run_option_pricing"]

    # Create instance 
    heston1 = HestonModel(**heston_params)
    
    # Create output directory
    if do_timestamp:
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        output_directory = os.path.join(output_dir, f'{scheme}_T={T}_n={n}_init={init_value}_N={N}', timestamp)
    else:
        output_directory = os.path.join(output_dir, f'{scheme}_T={T}_n={n}_init={init_value}_N={N}')
    os.makedirs(output_directory, exist_ok=True)
    
    # Run simulation processes
    if run_simulation: 
        print_section(f'Initiating {scheme} scheme simulation with time horizon T={T} and discretisation parameter n={n}')
        heston1.simulate_model(init_value=init_value, T=T, n=n, N=N, output_directory=output_directory, scheme=scheme)
    if run_analysis:
        print_section(f'Analysing simulation results from {output_directory}...')
        plot_trajectory(output_directory)
    if run_option_pricing:
        print_section(f'Pricing option with strike price K={strike_price} and maturity T_M={maturity}...')
        price_option(output_directory, strike_price, maturity)

    print_section("Workflow Complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Heston model workflow based on configuration file.")
    parser.add_argument('--config', type=str, default='config.json', help="Path to the JSON configuration file.")
    args = parser.parse_args()

    main(args.config)
