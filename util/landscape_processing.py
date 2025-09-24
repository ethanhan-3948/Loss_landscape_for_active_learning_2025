from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_loss_landscapes(df):
    """
    Extracts and organizes loss landscapes from a DataFrame into a dictionary.

    This function processes a DataFrame containing various loss landscapes, 
    identified by column names that include the term 'loss'. It organizes 
    these columns into a dictionary where each key is a unique loss function 
    name (derived from the column names), and each value is a DataFrame 
    containing the relevant columns for that loss function.

    Parameters:
    df (pd.DataFrame): The input DataFrame with columns representing different 
                       loss landscapes. The first column is assumed to be an 
                       identifier or index.

    Returns:
    dict: A dictionary with loss function names as keys and DataFrames as values, 
          each containing the relevant columns for that loss function.
    """
    # Get all column names except the first one
    column_names = df.columns[1:]
    # Extract the part before 'loss' for each column name
    loss_functions = [name.split('loss')[0].rstrip('_') for name in column_names]
    # Create a dictionary with loss functions as keys and DataFrames as values
    loss_function_dict = {}
    for loss_function in set(loss_functions):
        # Filter columns that correspond to the current loss function
        relevant_columns = [col for col in df.columns if col.startswith(loss_function)]
        # Create a new DataFrame with the first column and relevant columns
        loss_function_dict[loss_function] = df[[df.columns[0]] + relevant_columns]
    # Print the loss functions
    print(f"The loss functions used are: {', '.join(loss_function_dict.keys())}")
    return loss_function_dict

def apply_function_to_column_and_add(df, func, column_name, new_column_name, loss_type):
    """
    Applies a given function to each element in a specified column of a DataFrame
    and adds a new column with the results.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    func (callable): The function to apply to each element in the specified column.
    column_name (str): The name of the column to which the function will be applied.
    new_column_name (str): The name of the new column to be added to the DataFrame.
    loss_type (str): The prefix for the column names related to the loss type.

    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    # Apply the function to the specified column and store the results in a new column
    df = df.copy()  # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df.loc[:, loss_type + '_' + new_column_name] = df[loss_type + '_' + column_name].apply(func)
    return df



# 3. original loss
def loss_at_origin(np_array):
    """
    Returns the number at the center position of a 2D numpy array.

    Parameters:
    np_array (np.ndarray): The input numpy array with shape (n, n, 1), where n is even.

    Returns:
    float: The number at the center position.
    """
    # Remove the last dimension to work with a 2D array
    grid = np_array[:, :, 0]
    
    # Calculate the center index
    center_index = grid.shape[0] // 2
    
    # Get the number at the center position
    return grid[center_index, center_index]


# 4. avg loss
def average_loss(np_array):
    """
    Calculates the average of a given numpy array.

    Parameters:
    np_array (np.ndarray): The input numpy array.

    Returns:
    float: The average of the array elements.
    """
    return np_array.mean()


# 5. standard dev of loss
def standard_deviation_of_loss(np_array):
    """
    Calculates the standard deviation of a given numpy array.

    Parameters:
    np_array (np.ndarray): The input numpy array.

    Returns:
    float: The standard deviation of the array elements.
    """
    return np_array.std()


# 6. is original loss the lowest?
def is_original_loss_the_lowest(np_array):
    """
    Checks if the number at the center position of a 2D numpy array is the same as the lowest number.

    Parameters:
    np_array (np.ndarray): The input numpy array with shape (n, n, 1), where n is even.

    Returns:
    bool: True if the number at the center position is the same as the lowest number, False otherwise.
    """
    # Remove the last dimension to work with a 2D array
    grid = np_array[:, :, 0]
    
    # Calculate the center index
    center_index = grid.shape[0] // 2
    
    # Get the number at the center position
    middle_number = grid[center_index, center_index]
    
    # Find the lowest number in the grid
    lowest_number = grid.min()
    
    return middle_number == lowest_number


# 7. original/ second best
def lowest_over_original_loss(np_array, eps = 1e-10):
    """
    Calculates the ratio of the lowest number to the number at the center position in a 2D numpy array.

    Parameters:
    np_array (np.ndarray): The input numpy array with shape (n, n, 1), where n is even.

    Returns:
    float: The ratio of the lowest number to the number at the center position.
    """
    # Remove the last dimension to work with a 2D array
    grid = np_array[:, :, 0]
    
    # Calculate the center index
    center_index = grid.shape[0] // 2
    
    # Get the number at the center position
    middle_number = grid[center_index, center_index]
    
    # Find the lowest number in the grid
    lowest_number = grid.min()
    
    return lowest_number / middle_number + eps


# 8. distance between best and origin
def euclidean_distance_best_to_original(np_array):
    """
    Finds the location of the minimum value in a 2D grid numpy array and returns the Euclidean distance from it to the original model location (15, 15).

    Parameters:
    np_array (np.ndarray): The input numpy array with shape (30, 30, 1).

    Returns:
    float: The Euclidean distance from the location of the minimum value to the original model location (15, 15).
    """
    # Remove the last dimension to work with a 2D array
    grid = np_array[:, :, 0]
    
    # Find the index of the minimum value in the grid
    min_index = np.unravel_index(grid.argmin(), grid.shape)
    
    # Calculate the Euclidean distance from the min_index to the original model location (15, 15)
    distance = np.sqrt((min_index[0] - 15) ** 2 + (min_index[1] - 15) ** 2)
    
    return distance

def log_of_array(np_array, eps = 1e-10):
    """
    Returns the natural logarithm of each element in a numpy array.

    Parameters:
    np_array (np.ndarray): The input numpy array.

    Returns:
    np.ndarray: A numpy array with the natural logarithm of each element.
    """
    return np.log(np_array + eps)

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


def z_transform_standardize(df, column_name, loss_type):
    """
    Standardizes the (n, n, 1) shaped arrays in the specified column of the dataframe,
    and restores them to their original shape.

    Parameters:
    df (pd.DataFrame): The dataframe containing the column with (n, n, 1) shaped arrays.
    column_name (str): The name of the column to process.

    Returns:
    pd.DataFrame: A new dataframe with an additional column containing the standardized square arrays.
    """
    # Flatten each (n, n, 1) array to a 1D array
    flattened_arrays = np.array([array.flatten() for array in df[loss_type + '_' + column_name]])

    # Standardize the flattened arrays
    scaler = StandardScaler()
    standardized_flattened_arrays = scaler.fit_transform(flattened_arrays)

    # Restore the standardized arrays to their original (n, n, 1) shape
    standardized_square_arrays = [restore_to_square_shape(arr) for arr in standardized_flattened_arrays]

    # Add the standardized square arrays as a new column in the dataframe
    df[f'z_transformed_{loss_type}_{column_name}'] = standardized_square_arrays

    return df

def min_max_normalize(df, column_name, loss_type):
    """
    Min-max normalizes the (n, n, 1) shaped arrays in the specified column of the dataframe,
    and restores them to their original shape.

    Parameters:
    df (pd.DataFrame): The dataframe containing the column with (n, n, 1) shaped arrays.
    column_name (str): The name of the column to process.

    Returns:
    pd.DataFrame: A new dataframe with an additional column containing the normalized square arrays.
    """
    # Flatten each (n, n, 1) array to a 1D array
    flattened_arrays = np.array([array.flatten() for array in df[loss_type + '_' + column_name]])

    # Min-max normalize the flattened arrays
    min_vals = flattened_arrays.min(axis=1, keepdims=True)
    max_vals = flattened_arrays.max(axis=1, keepdims=True)
    normalized_flattened_arrays = (flattened_arrays - min_vals) / (max_vals - min_vals)

    # Restore the normalized arrays to their original (n, n, 1) shape
    normalized_square_arrays = [restore_to_square_shape(arr) for arr in normalized_flattened_arrays]

    # Add the normalized square arrays as a new column in the dataframe
    df[f'min_max_normalized_{loss_type}_{column_name}'] = normalized_square_arrays

    return df


def compute_wsd(p, q, eps=1e-10):
    """
    Compute the Wasserstein Distance (WSD) between two probability distributions.

    Parameters:
    p (np.ndarray): The first probability distribution array, shaped (n, n, 1).
    q (np.ndarray): The second probability distribution array, shaped (n, n, 1).
    eps (float): A small epsilon value to avoid division by zero issues. Default is 1e-10.

    Returns:
    float: The WSD value.
    """
    # Flatten the arrays and ensure they are probability distributions
    p = p.flatten()
    q = q.flatten()
    p /= p.sum()
    q /= q.sum()
    
    # Clip small eps to avoid division by zero issues
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    
    # Compute WSD
    wsd = wasserstein_distance(p, q)
    return wsd

def compute_jsd(p, q, eps=1e-10):
    """
    Compute the Jensen-Shannon Divergence (JSD) between two probability distributions.

    Parameters:
    p (np.ndarray): The first probability distribution array, shaped (n, n, 1).
    q (np.ndarray): The second probability distribution array, shaped (n, n, 1).
    eps (float): A small epsilon value to avoid log(0) issues. Default is 1e-10.

    Returns:
    float: The JSD value.
    """
    # Flatten the arrays and ensure they are probability distributions
    p = p.flatten()
    q = q.flatten()
    p /= p.sum()
    q /= q.sum()
    
    # Clip small eps to avoid log(0) issues
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    
    # Compute JSD
    jsd = jensenshannon(p, q)
    return jsd


def evaluate_pair_and_add_column(df1, df2, column_name, loss_type1, loss_type2, eval_function, additional_input):
    """
    Evaluate a function on pairs of arrays from two dataframes and add the result as a new column.

    Parameters:
    df1 (pd.DataFrame): The first dataframe.
    df2 (pd.DataFrame): The second dataframe.
    column_name (str): The base column name to concatenate with loss types.
    loss_type1 (str): The first loss type to concatenate with the column name.
    loss_type2 (str): The second loss type to concatenate with the column name.
    eval_function (callable): The function to evaluate on the pair of arrays.
    additional_input (str): The additional input string to form the new column name.

    Returns:
    None: The function modifies the dataframes in place.
    """
    # Construct the full column names
    col1 = f"{loss_type1}_{column_name}"
    col2 = f"{loss_type2}_{column_name}"
    
    # Construct the new column name
    new_col_name = f"{loss_type1}_vs_{loss_type2}_{additional_input}"
    
    # Evaluate the function on each pair of arrays and add the result to the new column
    df1[new_col_name] = df1.apply(lambda row: eval_function(row[col1], df2.loc[row.name, col2]), axis=1)
    df2[new_col_name] = df2.apply(lambda row: eval_function(df1.loc[row.name, col1], row[col2]), axis=1)

