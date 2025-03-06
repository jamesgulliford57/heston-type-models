import configparser
from utils.build_utils import parse_value, parse_possible_list, create_directory, load_class
from utils.sim_utils import print_section
from analysis import plot_trajectory, price_option

def main(config_file):
    """
    Main function runs workflow based on an INI configuration file.

    Parameters
    ---
    config_file : str
        Path to the INI config file containing simulation parameters and options.
    """
    # Load configuration from INI file
    config = configparser.ConfigParser()
    config.read(config_file)
    # Model
    model_name = config.get("model", "model_name")
    ModelClass = load_class(model_name) 
    # Model parameters 
    model_params = {key: parse_value(config.get("model_params", key)) 
                    for key in config.options("model_params")}
    # Simulation parameters
    init_value = config.get("simulation", "init_value")
    init_value = parse_possible_list(init_value)
    
    final_time = config.getfloat("simulation", "final_time")
    n = config.getint("simulation", "n")
    num_paths = config.getint("simulation", "num_paths")
    scheme = config.get("simulation", "scheme")
    # Output parameters
    output_dir = config.get("output", "output_dir")
    do_timestamp = config.getboolean("output", "do_timestamp")
    # Workflow controls
    run_simulation = config.getboolean("workflow", "run_simulation")
    run_analysis = config.getboolean("workflow", "run_analysis")
    # Option pricing parameters
    run_option_pricing = config.getboolean("option", "run_option_pricing")
    strike_price = config.getfloat("option", "strike_price")
    maturity = config.getfloat("option", "maturity")
    
    # Create output directory
    output_directory = create_directory(output_dir, model_name, scheme, final_time, n, init_value, num_paths, do_timestamp)
    # Instantiate model dynamically
    model = ModelClass(**model_params)
    
    # Run simulation processes
    if run_simulation: 
        print_section(f'Initiating {scheme} scheme simulation of {model_name} model with {num_paths} paths final_time={final_time} and discretisation parameter n={n}')
        model.simulate_model(init_value=init_value, final_time=final_time, n=n, num_paths=num_paths, output_directory=output_directory, scheme=scheme)
    if run_analysis:
        print_section(f'Analysing simulation results from {output_directory}...')
        plot_trajectory(output_directory)
    if run_option_pricing:
        print_section(f'Pricing option with strike price K={strike_price} and maturity T_M={maturity}...')
        price_option(output_directory, strike_price, maturity)
    
    print_section("Workflow Complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run model workflow based on configuration file.")
    parser.add_argument('--config', type=str, default='config_files/config.ini', help="Path to the INI configuration file.")
    args = parser.parse_args()
    main(args.config)
