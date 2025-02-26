import numpy as np

def write_npy(output_directory, **data_arrays):
        """
        Write simulation data to npy files to provided output directory.

        Parameters
        ---
        output_directory : str
            Path to directory where data will be saved.
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

        Parameters
        ---
        output_directory : str
            Path to directory where data will be saved.
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

def print_section(message, empty_lines_before=1, empty_lines_after=1):
    """
    Print a formatted section with separators and optional empty lines.

    Parameters
    ---
    message : str
        Message to be displayed in the section.
    empty_lines_before : int
        Number of empty lines to be printed before the section.
    empty_lines_after : int
        Number of empty lines to be printed after the section.
    """
    from colorama import Fore, Style
    
    separator_length=len(message)
    print("\n" * empty_lines_before, end="")  # Add empty lines before the section
    print(Fore.GREEN + "=" * separator_length)
    print(Fore.GREEN + message)
    print(Fore.GREEN + "=" * separator_length + Style.RESET_ALL)
    print("\n" * empty_lines_after, end="")  # Add empty lines after the section

def progress_bar_decorator(desc="Simulating", unit="step"):
    """
    Decorator that provides tqdm progress bar.

    Parameters
    ---
    desc : str
        Description of the progress bar.
    unit : str  
        Unit of the progress bar.
    """
    from tqdm import tqdm
    from functools import wraps

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

