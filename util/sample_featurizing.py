from jarvis.core.atoms import Atoms





def get_structure_dicts(df):
        """Extracts a list of structure dictionaries from the DataFrame."""
        return df['atoms'].apply(eval).tolist()

def create_atoms_instances(structure_dicts):
    """Creates a list of Atoms instances from structure dictionaries."""
    return [Atoms.from_dict(structure_dict) for structure_dict in structure_dicts]

def convert_to_pymatgen_structures(atoms_list):
    """Converts a list of Atoms instances to pymatgen structures."""
    return [atoms.pymatgen_converter() for atoms in atoms_list]

def get_pymatgen_structures_to_df(df):
    """Takes a DataFrame, processes it to add pymatgen structures, and returns the modified DataFrame."""
    try:
        atoms_instances = create_atoms_instances(df['atoms'])
    except:
        structure_dicts = get_structure_dicts(df)
        atoms_instances = create_atoms_instances(structure_dicts)
        
    pymatgen_structures = convert_to_pymatgen_structures(atoms_instances)
    # Add the pymatgen structures to the DataFrame
    df['pmg_structure'] = pymatgen_structures
    return df


import re

def check_element_presence(formula, element):
    """
    Checks for the presence of a specified element in the formula.

    Parameters:
    formula (str): The chemical formula to check.
    element (str): The element to check for in the formula.

    Returns:
    bool: True if the element is present, False otherwise.
    """
    # Remove all numbers from the formula
    formula_no_numbers = re.sub(r'\d+', '', formula)
    # Split by capital letters to get individual elements
    elements = re.findall(r'[A-Z][a-z]*', formula_no_numbers)
    # Check for exact equivalence with the input element
    return element in elements

def add_element_presence_column(df, element):
    """
    Adds a column to the DataFrame indicating the presence of a specified element in the formula.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'formula' column.
    element (str): The element to check for in the formula.

    Returns:
    pd.DataFrame: A new DataFrame with an additional column 'has_<element>' indicating presence.
    """
    df_copy = df.copy()
    column_name = f'has_{element}'
    df_copy[column_name] = df_copy['formula'].apply(lambda formula: check_element_presence(formula, element))
    return df_copy

def count_elements(formula):
    """
    Counts the number of different elements in a chemical formula.

    Parameters:
    formula (str): The chemical formula to analyze.

    Returns:
    int: The number of different elements in the formula.
    """
    # Use regex to find all unique elements in the formula
    elements = re.findall(r'[A-Z][a-z]*', formula)
    return len(set(elements))

def add_num_elements_column(df):
    """
    Adds a column to the DataFrame indicating the number of different elements in the formula.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'formula' column.

    Returns:
    pd.DataFrame: A new DataFrame with an additional column 'num_elements' indicating the number of different elements.
    """
    df_copy = df.copy()
    # Apply the count_elements function to each formula in the DataFrame
    df_copy['num_elements'] = df_copy['formula'].apply(count_elements)
    return df_copy