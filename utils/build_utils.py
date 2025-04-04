def parse_value(value):
    """
    Attempt to convert string value to corresponding Python literal.
    If the conversion fails the original string is returned.
    """
    import ast
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    
def parse_possible_list(value):
    """
    Parse input that is either a list or a single int or float.
    """
    import ast
    if not isinstance(value, str):
        return [value]

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except Exception:
        pass

    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        items = inner.split(',')
        parsed_items = []
        for item in items:
            item = item.strip()
            if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                item = item[1:-1].strip()
                parsed_items.append(item)
            else:
                try:
                    parsed_items.append(float(item))
                except ValueError:
                    parsed_items.append(item)
        return parsed_items

    return [value]

def to_camel_case(s):
    """
    Convert a snake_case string to CamelCase.
    
    For example, "black_scholes" becomes "BlackScholes".
    """
    return ''.join(word.capitalize() for word in s.split('_'))

def create_directory(model_name, simulator_name, simulator_params={}, do_timestamp=False):
    """
    Create output directory for simulation results.

    Parameters
    ---
    model_name : str
        Name of the model being simualted.
    scheme : str
        Numerical scheme used to simulate model.
    final_time : float
        Time horizon of simulation.
    n : int
        Discretisation parameter, number of intervals T is divided into.
    init_value : float
        Initial condition for random solution trajectories.
    num_paths : int
        Number of trajectories to be simulated.
    do_timestamp : bool
        Flag to include timestamp in output directory name.
    """

    import datetime
    import os 

    final_time = simulator_params['final_time']
    discretisation_parameter = simulator_params['discretisation_parameter']
    initial_value = simulator_params['initial_value']
    number_of_paths = simulator_params['number_of_paths']

    if do_timestamp:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        output_directory = os.path.join('data', model_name, simulator_name, f'T={final_time}_n={discretisation_parameter}_init={initial_value}_paths={number_of_paths}', timestamp)
    else:
        output_directory = os.path.join('data', model_name, simulator_name, f'T={final_time}_n={discretisation_parameter}_init={initial_value}_paths={number_of_paths}')
    os.makedirs(output_directory, exist_ok=True)

    return output_directory

def load_class(model_name):
    """
    Dynamically load class based on provided model name.
    
    Parameters
    ---
    model_name : str
        The name of the model.
    """
    import importlib

    module_name = model_name.lower()  
    class_name = to_camel_case(model_name) 
    module = importlib.import_module(f"models.{module_name}")
    return getattr(module, class_name)

def list_files_excluding(directory, exclude_files=None):
    """
    List files in a directory excluding those in the exclude_files list.

    Parameters
    ---
    directory : str
        The directory to list files from.
    exclude_files : list
        List of files to exclude from the listing.
    """
    import os
    if exclude_files is None:
        exclude_files = []
    elif not isinstance(exclude_files, list):
        exclude_files = [exclude_files]
    exclude_files.append('__pycache__')
    return [f for f in os.listdir(directory) if f not in exclude_files]