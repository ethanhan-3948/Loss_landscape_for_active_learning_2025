#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pickle
import yaml
import sys
from typing import Dict, Any

from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
    IonProperty,
    BandCenter
)
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures

from util.sample_featurizing import get_pymatgen_structures_to_df, add_num_elements_column
from util.landscape_processing import (
    extract_loss_landscapes,
    apply_function_to_column_and_add,
    loss_at_origin,
    average_loss,
    standard_deviation_of_loss,
    is_original_loss_the_lowest,
    lowest_over_original_loss,
    euclidean_distance_best_to_original,
    log_of_array,
    z_transform_standardize,
    min_max_normalize
)

def get_yml_pkl_files_paths(folder: str) -> Dict[str, str]:
    """Get paths to YAML and pickle files in the specified folder.
    
    Args:
        folder: Path to the folder containing the files
        
    Returns:
        Dictionary with paths to YAML and pickle files
    """
    files = os.listdir(folder)
    result = {'yml': None, 'pkl': None}
    
    for f in files:
        full_path = os.path.join(folder, f)
        if f.endswith('.yml'):
            result['yml'] = full_path
        elif f.endswith('.pkl'):
            result['pkl'] = full_path
            
    return result

def process_composition_features(sample_df: pd.DataFrame, n_jobs: int = 8, featurize: bool = False) -> pd.DataFrame:
    """Process composition features using matminer featurizers.
    
    Args:
        sample_df: DataFrame containing sample data
        n_jobs: Number of parallel jobs for featurization
        
    Returns:
        DataFrame with composition features
    """
    sample_composition_df = sample_df[['jid', 'formula']].copy()
    
    convert_featurizer = StrToComposition()
    convert_featurizer.set_n_jobs(n_jobs)
    sample_composition_df = convert_featurizer.featurize_dataframe(
        sample_composition_df, 'formula', ignore_errors=True
    )
    
    composition_featurizers = [
        ElementProperty.from_preset("magpie"),
        Stoichiometry(),
        ValenceOrbital(),
        IonProperty(),
        BandCenter()
    ]

    if featurize:
        for composition_featurizer in composition_featurizers:
            composition_featurizer.set_n_jobs(n_jobs)
            sample_composition_df = composition_featurizer.featurize_dataframe(
                sample_composition_df, 'composition', ignore_errors=True
            )
    
    return sample_composition_df

def process_structure_features(sample_df: pd.DataFrame, n_jobs: int = 8, featurize: bool = False) -> pd.DataFrame:
    """Process structure features using matminer featurizers.
    
    Args:
        sample_df: DataFrame containing sample data
        n_jobs: Number of parallel jobs for featurization
        
    Returns:
        DataFrame with structure features
    """
    sample_structure_df = sample_df[['jid', 'atoms']].copy()
    sample_structure_df = get_pymatgen_structures_to_df(sample_structure_df)
    
    structure_featurizers = [DensityFeatures(), GlobalSymmetryFeatures()]
    
    if featurize:
        for structure_featurizer in structure_featurizers:
            structure_featurizer.set_n_jobs(n_jobs)
            sample_structure_df = structure_featurizer.featurize_dataframe(
                sample_structure_df, 'pmg_structure', ignore_errors=True
            )
    
    return sample_structure_df

def process_loss_landscapes(loss_landscapes_df: pd.DataFrame, run_id: str) -> Dict[str, pd.DataFrame]:
    """Process loss landscapes and compute various metrics.
    
    Args:
        loss_landscapes_df: DataFrame containing loss landscapes
        run_id: Identifier for the current run
        
    Returns:
        Dictionary containing processed loss landscape data
    """
    loss_landscapes_df = loss_landscapes_df.rename(
        columns={'raw_loss_landscapes': f'{run_id}_mse_loss_landscape_array'}
    )
    
    loss_function_dict = extract_loss_landscapes(loss_landscapes_df)
    
    for loss_function, df in loss_function_dict.items():
        metrics = [
            (loss_at_origin, "loss_at_origin"),
            (average_loss, "average_of_landscape"),
            (standard_deviation_of_loss, "std_dev_of_landscape"),
            (is_original_loss_the_lowest, "is_original_loss_lowest"),
            (lowest_over_original_loss, "lowest_loss_over_original"),
            (euclidean_distance_best_to_original, "euclidean_distance_best_to_original"),
            (log_of_array, "log_loss_landscape_array")
        ]
        
        for metric_func, column_name in metrics:
            df = apply_function_to_column_and_add(
                df, metric_func, "loss_landscape_array", column_name, loss_function
            )
        
        df = z_transform_standardize(df, "log_loss_landscape_array", loss_function)
        df = min_max_normalize(df, "log_loss_landscape_array", loss_function)
        
        loss_function_dict[loss_function] = df
    
    return loss_function_dict

def save_results(folder: str, sample_df: pd.DataFrame, 
                sample_composition_df: pd.DataFrame,
                sample_structure_df: pd.DataFrame,
                loss_function_dict: Dict[str, pd.DataFrame]) -> None:
    """Save all processed results to pickle files.
    
    Args:
        folder: Output folder path
        sample_df: DataFrame with sample data
        sample_composition_df: DataFrame with composition features
        sample_structure_df: DataFrame with structure features
        loss_function_dict: Dictionary with processed loss landscapes
    """
    output_files = {
        'feat_sample_df.pkl': sample_df,
        'feat_sample_composition_df.pkl': sample_composition_df,
        'feat_sample_structure_df.pkl': sample_structure_df,
        'processed_loss_function_dict.pkl': loss_function_dict
    }
    
    for filename, data in output_files.items():
        with open(os.path.join(folder, filename), 'wb') as f:
            pickle.dump(data, f)

def main(config_path: str) -> None:
    """Main function to process loss landscapes.
    
    Args:
        config_path: Path to the configuration YAML file
    """
    print("\n=== Starting Loss Landscape Processing ===")
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    folder = config['folder']
    featurize = config.get('featurize', 'False') == 'True'
    n_jobs = config.get('n_jobs', 8)
    print(f"Processing folder: {folder}")
    print(f"Using {n_jobs} parallel jobs for featurization")
    
    # Get file paths
    print("\nLocating YAML and pickle files...")
    extracted_files_paths = get_yml_pkl_files_paths(folder)
    print(f"Found config: {extracted_files_paths['yml']}")
    print(f"Found data: {extracted_files_paths['pkl']}")
    
    # Load experiment config
    print("\nLoading experiment configuration...")
    with open(extracted_files_paths['yml'], 'r') as f:
        expt_config = yaml.safe_load(f)
    
    # Load data
    print("\nLoading sample and loss landscape data...")
    sample_df = pd.read_pickle(expt_config['data_path'])
    loss_landscapes_df = pd.read_pickle(extracted_files_paths['pkl'])
    run_id = expt_config['run_id']
    print(f"Loaded {len(sample_df)} samples")
    print(f"Run ID: {run_id}")
    
    # Process sample data
    print("\nProcessing sample data...")
    sample_df = add_num_elements_column(sample_df)
    
    # Process features
    print("\nProcessing composition features...")
    sample_composition_df = process_composition_features(sample_df, n_jobs, featurize)
    print(f"Generated {len(sample_composition_df.columns)} composition features")
    
    print("\nProcessing structure features...")
    sample_structure_df = process_structure_features(sample_df, n_jobs, featurize)
    print(f"Generated {len(sample_structure_df.columns)} structure features")
    
    # Process loss landscapes
    print("\nProcessing loss landscapes...")
    loss_function_dict = process_loss_landscapes(loss_landscapes_df, run_id)
    print(f"Processed {len(loss_function_dict)} loss functions")
    
    # Save results
    print("\nSaving results...")
    save_results(
        folder,
        sample_df,
        sample_composition_df,
        sample_structure_df,
        loss_function_dict
    )
    print("\n=== Loss Landscape Processing Complete ===\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python post_process_loss_landscapes.py <config_file>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    main(config_path)