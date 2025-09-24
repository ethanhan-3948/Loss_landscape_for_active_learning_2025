import numpy as np
from ipywidgets import Dropdown
from IPython.display import display
from pymatgen.core import Structure
from jarvis.core.atoms import pmg_to_atoms


def restore_to_square_shape(flattened_array):
    """
    Restores a flattened array to its original shape of (n, n, 1).

    Parameters:
    flattened_array (list or np.ndarray): The flattened array to be reshaped.

    Returns:
    np.ndarray: The reshaped array with shape (n, n, 1).
    """
    # Convert the flattened array to a numpy array if it isn't already
    flattened_array = np.array(flattened_array)
    
    # Calculate n from the length of the flattened array
    n = int(np.sqrt(flattened_array.size))
    
    # Reshape the array to (n, n, 1)
    original_shape_array = flattened_array.reshape((n, n, 1))
    
    return original_shape_array

def check_nan(df):
    # Print the shape of the DataFrame
    print(f"Shape of the DataFrame: {df.shape}")
    
    # Calculate the total number of NaN entries
    total_nan_entries = df.isnull().sum().sum()
    print(f"Total number of NaN entries: {total_nan_entries}")
    print()
    
    # Analyze columns with NaN values
    nan_per_column = df.isnull().sum()
    columns_with_nan = nan_per_column[nan_per_column > 0]
    print(f"Number of columns with NaN values: {len(columns_with_nan)}")
    for column, nan_count in columns_with_nan.items():
        column_nan_percentage = (nan_count / df.shape[0]) * 100
        print(f"Column '{column}' has {nan_count} NaN values ({column_nan_percentage:.2f}%)")
    
    print()

    # Analyze rows with NaN values
    rows_with_nan = df.isnull().any(axis=1)
    num_rows_with_nan = rows_with_nan.sum()
    print(f"Number of rows with NaN values: {num_rows_with_nan}")
    for index, row in df[rows_with_nan].iterrows():
        nan_count = row.isnull().sum()
        row_nan_percentage = (nan_count / df.shape[1]) * 100
        print(f"Row {index} (first entry: {row.iloc[0]}) has {nan_count} NaN values ({row_nan_percentage:.2f}%)")

def flatten_and_vstack(data_column):
    """Flattens each (n, n, 1) array in the column and vertically stacks them."""
    return np.vstack([np.ravel(arr) for arr in data_column])

def create_selectors(sample_dict, loss_function_dict):
    """Creates and returns dropdown selectors for sample data and loss functions
    
    Args:
        sample_dict: Dictionary containing sample DataFrames
        loss_function_dict: Dictionary containing loss function DataFrames
        
    Returns:
        dict: Dictionary containing all selector widgets
    """
    style = {'description_width': 'initial'}
    
    # First set of property selectors
    combined_dict_1 = {**sample_dict, **loss_function_dict}
    property_dict_selector_1 = Dropdown(
        options=list(combined_dict_1.keys()),
        value=list(combined_dict_1.keys())[0],
        description='Select DataFrame 1',
        style=style,
        disabled=False
    )

    property_column_selector_1 = Dropdown(
        options=combined_dict_1[property_dict_selector_1.value].columns.tolist(),
        description='Select Property 1',
        style=style,
        disabled=False
    )

    def update_property_columns_1(*args):
        property_column_selector_1.options = combined_dict_1[property_dict_selector_1.value].columns.tolist()

    property_dict_selector_1.observe(update_property_columns_1, names='value')

    # Second set of property selectors
    combined_dict_2 = {**sample_dict, **loss_function_dict}
    property_dict_selector_2 = Dropdown(
        options=list(combined_dict_2.keys()),
        value=list(combined_dict_2.keys())[0],
        description='Select DataFrame 2',
        style=style,
        disabled=False
    )

    property_column_selector_2 = Dropdown(
        options=combined_dict_2[property_dict_selector_2.value].columns.tolist(),
        description='Select Property 2',
        style=style,
        disabled=False
    )

    def update_property_columns_2(*args):
        property_column_selector_2.options = combined_dict_2[property_dict_selector_2.value].columns.tolist()

    property_dict_selector_2.observe(update_property_columns_2, names='value')

    # First set of loss function selectors
    loss_function_selector_1 = Dropdown(
        options=list(loss_function_dict.keys()),
        value=list(loss_function_dict.keys())[0],
        description='Select Loss Function 1',
        style=style,
        disabled=False
    )

    loss_function_column_selector_1 = Dropdown(
        options=loss_function_dict[loss_function_selector_1.value].columns.tolist(),
        description='Select Landscape 1',
        style=style,
        disabled=False
    )

    def update_loss_function_columns_1(*args):
        loss_function_column_selector_1.options = loss_function_dict[loss_function_selector_1.value].columns.tolist()

    loss_function_selector_1.observe(update_loss_function_columns_1, names='value')

    # Second set of loss function selectors
    loss_function_selector_2 = Dropdown(
        options=list(loss_function_dict.keys()),
        value=list(loss_function_dict.keys())[0],
        description='Select Loss Function 2',
        style=style,
        disabled=False
    )

    loss_function_column_selector_2 = Dropdown(
        options=loss_function_dict[loss_function_selector_2.value].columns.tolist(),
        description='Select Landscape 2',
        style=style,
        disabled=False
    )

    def update_loss_function_columns_2(*args):
        loss_function_column_selector_2.options = loss_function_dict[loss_function_selector_2.value].columns.tolist()

    loss_function_selector_2.observe(update_loss_function_columns_2, names='value')

    # Return dictionary of all selectors
    selectors = {
        'property_dict_selector_1': property_dict_selector_1,
        'property_column_selector_1': property_column_selector_1,
        'property_dict_selector_2': property_dict_selector_2,
        'property_column_selector_2': property_column_selector_2,
        'loss_function_selector_1': loss_function_selector_1,
        'loss_function_column_selector_1': loss_function_column_selector_1,
        'loss_function_selector_2': loss_function_selector_2,
        'loss_function_column_selector_2': loss_function_column_selector_2
    }

    # Display selectors
    for selector in selectors.values():
        display(selector)

    return selectors

def add_atoms_structure(df, column_name):
    """
    Given a DataFrame and the name of a column containing pymatgen structure dictionaries,
    creates a new column 'atoms' containing the corresponding JARVIS Atoms objects (as dicts).

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column containing pymatgen structure dictionaries.

    Returns:
    pd.DataFrame: A copy of the DataFrame with an additional 'atoms' column containing JARVIS Atoms dicts.
    """
    df_copy = df.copy()

    def create_atoms_from_dict(structure_dict):
        structure = Structure.from_dict(structure_dict)
        atoms_structure = pmg_to_atoms(structure)
        atoms_dict = atoms_structure.to_dict()
        return atoms_dict

    df_copy['atoms'] = df_copy[column_name].apply(create_atoms_from_dict)
    return df_copy