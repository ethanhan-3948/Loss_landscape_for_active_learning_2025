import json
import torch
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from collections import OrderedDict
from torchinfo import summary
import pandas as pd
from util.utils_AD import *
import seaborn as sns
import numpy as np
import loss_landscapes
import loss_landscapes.metrics
from loss_landscapes.model_interface.model_wrapper import ModelWrapper
from abc import ABC, abstractmethod
import yaml
import argparse
import os
import copy

class Metric(ABC):
    """ A quantity that can be computed given a model or an agent. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, model_wrapper: ModelWrapper):
        pass

class Loss(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, loss_fn, model, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.model = model
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        outputs = model_wrapper.forward(self.inputs)
        err = self.loss_fn(self.target[0], outputs)
        return err

def split_3d_array(array):
    """Split a NxNx3 array into list of three NxNx1 arrays"""
    return [array[:,:,i:i+1] for i in range(array.shape[2])]

def generate_loss_landscapes(config):
    # Set device
    device = 'cuda' if torch.cuda.is_available() and config['device']=='cuda' else 'cpu'
    print(f"Using device: {device}")

    # Set paths
    eigenvector_max_path = os.path.join(os.path.normpath(config['eigenvector_folder_path']).replace('\\', '/'), 'model_eig_max.pt')
    eigenvector_min_path = os.path.join(os.path.normpath(config['eigenvector_folder_path']).replace('\\', '/'), 'model_eig_min.pt')

    # Load model and weights
    checkpoint = torch.load(os.path.normpath(config['model_path'].replace('\\', '/')), map_location=torch.device(device), weights_only=False)
    model = ALIGNN()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model_wt_dict = OrderedDict([i for i in model.named_parameters()])

    # Set up model for loss landscape calculation
    loss_func = torch.nn.MSELoss()
    func = copy.deepcopy(model)
    func.to(device)
    print(f"func device: {device}")
    func.eval()

    og_params = [i[1] for i in func.named_parameters() if len(i[1].size()) >= 1]
    og_layer_names = [i[0] for i in func.named_parameters() if len(i[1].size())>=1]

    # Load data
    data_df = pd.read_pickle(os.path.normpath(config['data_path']).replace('\\', '/'))
    target = config['target']
    data_list = [row.to_dict() for _, row in data_df.iterrows()]
    data_loader = get_data_loader(data_list, target, workers=0)

    # Load eigenvectors
    model_eig_max = copy.deepcopy(func)
    model_eig_max.load_state_dict(torch.load(eigenvector_max_path, weights_only=True))
    model_eig_min = copy.deepcopy(func)
    model_eig_min.load_state_dict(torch.load(eigenvector_min_path, weights_only=True))

    # Move models to device
    func.to(device)
    model_eig_max.to(device)
    model_eig_min.to(device)

    # Create metrics for each batch
    metric_list = []
    for batch in data_loader:
        s0_device = batch[0].to(device)
        s1_device = batch[1].to(device)
        s2_device = batch[2].to(device)
        x_train = (s0_device, s1_device)
        y_train = (s2_device)
        metric_list.append(Loss(loss_func, func.eval(), x_train, y_train))

    #scale factor used to scale hessian eigenvector for perturbation
    try:
        scale_factor = config['scale_factor']
    except:
        scale_factor = 1

    #half is to speed up the computation, if True, skip every other loss computation
    try:
        half = config['half']
    except:
        half = False

    # Generate loss landscapes
    try:
        loss_data_fin = loss_landscapes.batch_planar_interpolation(
            model_start=func.eval(),
            model_end_one=model_eig_max.eval(),
            model_end_two=model_eig_min.eval(),
            metric_list=metric_list,
            steps=config['steps'],
            deepcopy_model=True,
            scale=scale_factor,
            half=half
        )
    except Exception as e:
        print(str(e))
        return

    # Process results
    landscapes_list = split_3d_array(loss_data_fin)

    # Create dataframe for saving
    loss_landscapes_df = pd.DataFrame()
    loss_landscapes_df['jid'] = data_df['jid']
    loss_landscapes_df['raw_loss_landscapes'] = landscapes_list

    # Create save directory
    save_path = os.path.join('computed_loss_landscapes', config['run_id'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created folder: {save_path}")
    else:
        print(f"Folder already exists: {save_path}")

    # Save results
    loss_landscapes_df.to_pickle(os.path.join(save_path, 'loss_landscapes_df.pkl'))
    print(f"Saved loss landscapes dataframe to: {os.path.join(save_path, 'loss_landscapes_df.pkl')}")

    np.save(os.path.join(save_path, 'raw_loss_landscape_array.npy'), loss_data_fin)
    print(f"Saved loss landscapes array to: {os.path.join(save_path, 'raw_loss_landscape_array.npy')}")

    with open(os.path.join(save_path, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    print(f"Saved config to: {os.path.join(save_path, 'config.yml')}")

def main():
    parser = argparse.ArgumentParser(description='Generate loss landscapes')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    generate_loss_landscapes(config)

if __name__ == "__main__":
    main()
