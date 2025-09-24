#util for loading and parsing data

import os
import numpy as np
import pandas as pd

def extract_loss_landscapes(folders):
    """
    Process each folder to find sub-folders containing 'JVASP' in their name,
    load .npy files, and return a DataFrame with JVASP id and the corresponding loss landscapes.

    Parameters:
    folders (dict): A dictionary where keys are identifiers and values are folder paths.

    Returns:
    pd.DataFrame: A DataFrame containing JVASP id and the corresponding loss landscapes for each key-path pair.
    """
    data = {}
    for key, path in folders.items():
        # Find all sub-folders within the given path
        sub_folders = find_sub_folders(path)
        for sub_folder in sub_folders:
            # Check if the sub-folder name contains 'JVASP'
            is_jvasp_folder = any('JVASP' in string for string in sub_folder.split('_'))
            if is_jvasp_folder:
                # Extract the JVASP id from the sub-folder name
                jvasp_id = next((part for part in sub_folder.split('_') if 'JVASP' in part), None)
                # Find all .npy files within the JVASP sub-folder
                npy_files = find_npy_files(sub_folder)
                for npy_file in npy_files:
                    # Load the .npy file into a numpy array
                    np_array = np.load(npy_file)
                    # Initialize the data dictionary for the JVASP id if not already present
                    if jvasp_id not in data:
                        data[jvasp_id] = {'jid': jvasp_id}
                    # Store the numpy array in the data dictionary under the key-specific loss landscape
                    data[jvasp_id][f'{key}_loss_landscape_array'] = np_array
    # Convert the data dictionary to a DataFrame and return it
    return pd.DataFrame(data.values())

def find_sub_folders(path):
    """Find all sub-folder paths under the given folder."""
    sub_folders = []
    for root, dirs, _ in os.walk(path):
        for dir_name in dirs:
            # Append the full path of each sub-folder to the list
            sub_folders.append(os.path.join(root, dir_name))
    return sub_folders

def find_npy_files(sub_folder_path):
    """Find all .npy files under the sub-folder."""
    npy_files = []
    for sub_root, _, sub_files in os.walk(sub_folder_path):
        for file_name in sub_files:
            # Check if the file has a .npy extension and add it to the list
            if file_name.endswith('.npy'):
                npy_files.append(os.path.join(sub_root, file_name))
    return npy_files

def load_sample_csvs(base_path):
    """
    Given a base path, this function creates paths to 'test_subset.csv' and 'train_subset.csv',
    reads these CSV files, concatenates them, and returns the concatenated DataFrame.
    
    Parameters:
    base_path (str): The base directory path where the 'test' and 'train' CSV files are located.
    
    Returns:
    pd.DataFrame: The concatenated DataFrame containing data from both 'test_subset.csv' and 'train_subset.csv'.
    """
    # Create paths for the test and train CSV files
    test_csv_path = os.path.join(base_path, 'test_subset.csv')
    train_csv_path = os.path.join(base_path, 'train_subset.csv')
    
    # Read the CSV files into DataFrames
    test_df = pd.read_csv(test_csv_path)
    train_df = pd.read_csv(train_csv_path)
    
    # Concatenate the DataFrames
    concatenated_df = pd.concat([test_df, train_df], ignore_index=True)
    
    return concatenated_df