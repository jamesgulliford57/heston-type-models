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
        from os.path import join
        from numpy import save

        # Save each array to its own file in the directory
        for array_name, array_data in data_arrays.items():
            file_path = join(output_directory, f"{array_name}.npy")
            save(file_path, array_data)

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
        from os.path import join
        from json import dump

        # Save each array to its own file in the directory
        for array_name, array_data in data_arrays.items():
            file_path = join(output_directory, f"{array_name}.json")
            with open(file_path, "w") as f:
                  dump(array_data, f)

        print(f"Data saved in {output_directory}")
    

def update_json(json_path, iat, eff_sample_size):
    import os 
    import json 
    
    if os.path.exists(json_path):
        with open(json_path, 'r+', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            # Update data
            data['iat'] = round(iat, 4)
            data['eff_sample_size'] = round(eff_sample_size)
            
            # Move file pointer to the start of the file
            f.seek(0)
            # Overwrite with updated JSON
            json.dump(data, f, indent=4)
            # Truncate in case the new content is shorter
            f.truncate()
    else:
        # If file doesn't exist, just create it
        data = {
            'iat': round(iat, 4),
            'eff_sample_size': round(eff_sample_size)
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    return data