import numpy as np
import torch
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from collections import OrderedDict
from torchinfo import summary
import pandas as pd
from util.utils_AD import *
from src.hessian_eigenvector import min_max_hessian_eigs, npvec_to_tensorlist, force_wts_into_model


def save_results(run_id, config, model_eig_max, model_eig_min, maxeig, mineig, second_maxeig):
    """Save Hessian eigenvectors and config to a directory.
    
    Args:
        run_id: String identifier for this run
        config: Dictionary of configuration parameters
        model_eig_max: Model with maximum eigenvector weights
        model_eig_min: Model with minimum eigenvector weights
    """
    import os
    import yaml
    
    # Create output directory
    save_dir = os.path.join('eigenvectors', run_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save models
    torch.save(model_eig_max.state_dict(), 
              os.path.join(save_dir, 'model_eig_max.pt'))
    torch.save(model_eig_min.state_dict(),
              os.path.join(save_dir, 'model_eig_min.pt'))
    
    # Save config
    config_path = os.path.join(save_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save eigenvalues
    eig_path = os.path.join(save_dir, 'eigenvalues.txt')
    with open(eig_path, 'w') as f:
        f.write(f"Maximum eigenvalue: {maxeig}\n")
        f.write(f"Minimum eigenvalue: {mineig}\n") 
        f.write(f"Second maximum eigenvalue: {second_maxeig}\n")
    
    print(f"✅ Results saved to {save_dir}")



def main(config):
    """Compute and save Hessian eigenvectors for a model.
    
    Args:
        config: Dictionary containing configuration with keys:
            - model_path: Path to model checkpoint
            - data_path: Path to data file
            - target: Target variable name
            - device: Device to use (cuda/cpu)
    """
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and weights
    checkpoint = torch.load(os.path.normpath(config['model_path']).replace('\\', '/'), map_location=torch.device(device), weights_only=False)
    model = ALIGNN()
    model.load_state_dict(checkpoint["model"]) 
    model.eval()

    model_wt_dict = OrderedDict([i for i in model.named_parameters()])

    # Load data
    data_df = pd.read_pickle(os.path.normpath(config['data_path']).replace('\\', '/'))
    target = config.get('target', 'dHf')
    data_list = [row.to_dict() for _, row in data_df.iterrows()]
    data_loader = get_data_loader(data_list, target, workers=0)

    # Hessian eigenvector computation
    loss_func = torch.nn.MSELoss()
    func = copy.deepcopy(model)

    func.to(device)
    print(f"func device: {device}")
    func.eval()

    og_params = [i[1] for i in func.named_parameters() if len(i[1].size()) >= 1]
    og_layer_names = [i[0] for i in func.named_parameters() if len(i[1].size())>=1]

    maxeig, mineig, maxeigvec, mineigvec, second_maxeig, second_maxeigvec = min_max_hessian_eigs(
        func, data_loader, loss_func, all_params=False, verbose=False, use_cuda=(device=='cuda')
    )


    # Convert eigenvectors to model weights
    max_model_wts = npvec_to_tensorlist(maxeigvec, og_params)
    min_model_wts = npvec_to_tensorlist(mineigvec, og_params)

    model_eig_max = copy.deepcopy(func)
    model_eig_min = copy.deepcopy(func)

    # Load eigenvectors into models
    model_eig_max = force_wts_into_model(og_layer_names, max_model_wts, model_eig_max, model_wt_dict)
    model_eig_min = force_wts_into_model(og_layer_names, min_model_wts, model_eig_min, model_wt_dict)

    # Save models
    save_results(
        run_id=config['run_id'],
        config=config,
        model_eig_max=model_eig_max,
        model_eig_min=model_eig_min,
        maxeig=maxeig,
        mineig=mineig,
        second_maxeig=second_maxeig
    )

if __name__ == '__main__':
    import yaml
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python hessian_eigenvector.py config.yml")
        sys.exit(1)
        
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)