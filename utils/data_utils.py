def write_npy(directory, **data_arrays):
        """
        Write simulation data to npy files.

        Parameters
        ----------
        directory : str
            Path to directory where data will be saved.
        **data_arrays: keyword arguments
            Each keyword corresponds to the name of the array, and the value is the array to be saved.
        """
        from os.path import join
        from numpy import save
        for array_name, array_data in data_arrays.items():
            file_path = join(directory, f"{array_name}.npy")
            save(file_path, array_data)
        print(f"{file_path} saved.")

def write_json(directory, **data_arrays):
        """
        Write simulation parameters to json.

        Parameters
        ----------
        directory : str
            Path to directory where data will be saved.
        **data_arrays: keyword arguments
            Each keyword corresponds to the name of the array, and the value is the array to be saved.
        """
        from os.path import join
        from json import dump
        for array_name, array_data in data_arrays.items():
            file_path = join(directory, f"{array_name}.json")
            with open(file_path, "w") as f:
                  dump(array_data, f, indent=4)
        print(f"{file_path} saved.")


def read_json(json_path):
    """
    Read data from a json file.

    Parameters
    ----------
    json_path : str
        Path to the json file.
    """
    import json
    import os
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                output = json.load(f)
        except json.JSONDecodeError:
            output = {}
    else:
        output = {}

    return output
