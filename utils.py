import numpy as np
from tqdm import tqdm
from functools import wraps

def write_npy(output_directory, **data_arrays):
        """
        Write simulation data with a timestamp for uniqueness.
        Parameters:
        folder_name: provide name of folder in format f"...{}..."
        **data_arrays: keyword arguments
            Each keyword corresponds to the name of the array, and the value is the array to be saved.
        """
        import os

        # Save each array to its own file in the directory
        for array_name, array_data in data_arrays.items():
            file_path = os.path.join(output_directory, f"{array_name}.npy")
            np.save(file_path, array_data)

        print(f"Data saved in {output_directory}")

def write_json(output_directory, **data_arrays):
        """
        Write simulation data with a timestamp for uniqueness.
        Parameters:
        folder_name: provide name of folder in format f"...{}..."
        **data_arrays: keyword arguments
            Each keyword corresponds to the name of the array, and the value is the array to be saved.
        """
        import os
        import json

        # Save each array to its own file in the directory
        for array_name, array_data in data_arrays.items():
            file_path = os.path.join(output_directory, f"{array_name}.json")
            with open(file_path, "w") as f:
                  json.dump(array_data, f)

        print(f"Data saved in {output_directory}")

def print_section(message, empty_lines_before=1, empty_lines_after=1, separator_length=30):
    """
    Print a formatted section with separators and optional empty lines.
    """
    from colorama import Fore, Style
    print("\n" * empty_lines_before, end="")  # Add empty lines before the section
    print(Fore.GREEN + "=" * separator_length)
    print(Fore.GREEN + message)
    print(Fore.GREEN + "=" * separator_length + Style.RESET_ALL)
    print("\n" * empty_lines_after, end="")  # Add empty lines after the section

def progress_bar_decorator(desc="Simulating", unit="step"):
    """
    Decorator that provides a tqdm progress bar in your function.
    
    Expects the wrapped function to have an `N` keyword argument
    (the total number of steps) and optionally accept a `pbar` keyword 
    argument (which will be the tqdm progress bar).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to grab 'N' from kwargs
            N = kwargs.get('N')
            if N is None:
                raise ValueError("The wrapped function must have an 'N' keyword argument.")
            
            # Create the progress bar
            with tqdm(total=N-1, desc=desc, unit=unit, ncols=60, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                # Insert the progress bar into the function kwargs
                kwargs['pbar'] = pbar
                # Call the actual function
                result = func(*args, **kwargs)
            
            return result
        return wrapper
    return decorator

