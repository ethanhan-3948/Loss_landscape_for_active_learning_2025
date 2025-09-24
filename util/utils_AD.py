import copy
import numpy as np
import matplotlib.pyplot as plt
import alignn
from alignn.pretrained import *
from jarvis.db.figshare import data
from jarvis.db.jsonutils import loadjson
from collections import OrderedDict

import ipywidgets as widgets
from torchinfo import summary

import loss_landscapes
import loss_landscapes.metrics

from abc import ABC, abstractmethod
from loss_landscapes.model_interface.model_wrapper import ModelWrapper
import torch
from pandas import DataFrame
import pandas as pd

def get_hessians(func, data_loader, model_wt_dict, loss_func):
    import src.botcher_hessian_alignn as hess
    from src.botcher_hessian_alignn import min_max_hessian_eigs
    from src import botcher_utilities as util

    list_of_all_parameters = list(model_wt_dict.keys())

    # og_params = [func.state_dict()[i] for i in list_of_all_parameters if func.state_dict()[i].shape[0] > 1]
    # og_layer_names = [i for i in list_of_all_parameters if func.state_dict()[i].shape[0] >1]

    # og_params = []
    # og_layer_names = []
    # for i in list_of_all_parameters:
    #     try:
    #         if model_wt_dict[i].shape[0]> 1:
    #             tmp = model_wt_dict[i]
    #             og_params.append(tmp)
    #             og_layer_names.append(i)
    #     except:
    #         if i.split('.')[-1] == 'num_batches_tracked':
    #             continue
    #         else:
    #             print(i)



    og_params = [i[1] for i in func.named_parameters() if len(i[1].size()) > 1]
    og_layer_names = [i[0] for i in func.named_parameters() if len(i[1].size())>1]

    # x_train = (single_batch[0], single_batch[1])
    # y_train = single_batch[2]

    maxeig, mineig, maxeigvec, mineigvec, num_iter = min_max_hessian_eigs(func, data_loader, loss_func, all_params=False)

    max_model_wts = hess.npvec_to_tensorlist(maxeigvec, og_params)
    min_model_wts = hess.npvec_to_tensorlist(mineigvec, og_params)
    
    model_eig_max = copy.deepcopy(func)
    model_eig_min = copy.deepcopy(func)

    model_eig_max = force_wts_into_model(og_layer_names, max_model_wts, model_eig_max, model_wt_dict)
    model_eig_min = force_wts_into_model(og_layer_names, min_model_wts, model_eig_min, model_wt_dict)

    return model_eig_max, model_eig_min

def get_config(cfg):
    import json
    with open(cfg, 'r') as file:
        model_cfg = json.load(file)

    config = loadjson(cfg)
    config = TrainingConfig(**config)
    return config

def plot_model_wt_dist(model):
    model_wts = []
    model_biases = []
    model.eval()
    for i, _ in model.named_parameters():
    # for i in model.state_dict().keys():
        if i.split('.')[-1] == 'num_batches_tracked':
            continue
        # print(i[0])
        values = model.state_dict()[i].detach().numpy().flatten()
        if np.ndim(values) == 0:
            values = np.expand_dims(values, axis=0)

        if i.split('.')[-1] == 'weight':
            model_wts.extend(list(values))
        else:
            model_biases.extend(list(values))
    num_nan_wts = np.count_nonzero(np.isnan(model_wts))
    num_nan_b = np.count_nonzero(np.isnan(model_biases))

    if num_nan_wts > 0 or num_nan_b > 0:
        print(f'There are {num_nan_wts} NaN Model Wts')
        print(f'There are {num_nan_b} NaN Model Biases')

    wt_ct, wt_bins = np.histogram([model_wts], bins=25)
    b_ct, b_bins = np.histogram([model_biases], bins=25)

    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    ax[0].hist(wt_bins[1:], wt_bins[1:], weights=wt_ct)
    ax[0].set_title('Weights')
    ax[0].set_xlabel('Value')
    ax[0].set_ylabel('Count')

    ax[1].hist(b_bins[1:], b_bins[1:], weights=b_ct)
    ax[1].set_title('Biases')
    ax[1].set_xlabel('Value')
    ax[1].set_ylabel('Count')
    plt.tight_layout()
    plt.show()

    print('avg param val: ', np.mean(model_wts+model_biases))
    print('std param val: ', np.std(model_wts+model_biases))


def load_model(config, model_name, print_summary=False):
    model = AlignnLayerNorm(config)
    model_wt_dict = torch.load(model_name, weights_only=False, map_location=device)['model']
    bad_keys = model.load_state_dict(model_wt_dict, strict=False)
    model.to(device)
    _ = model.eval()
    if print_summary:
        summary(model)

    return model, model_wt_dict

def load_pretrained_model(model_name, print_summary=False):
    model = get_figshare_model(model_name)
    model_wt_dict = model.state_dict()
    model.to(device)
    _ = model.eval()
    if print_summary:
        summary(model)
    return model, model_wt_dict



def load_jarvis_dft3d(n_samples):
    d = data("dft_3d")
    d = d[:n_samples]
    dataset = DataFrame(copy.deepcopy(d))
    atoms_df = DataFrame(list(DataFrame(d)['atoms']))
    dataset = pd.concat([dataset, atoms_df], axis=1)
    return dataset


def load_data(element_to_omit_from_training_data, n_samples):
    d = data("dft_3d")
    d = d[:n_samples]
    dataset = DataFrame(copy.deepcopy(d))
    atoms_df = DataFrame(list(DataFrame(d)['atoms']))
    dataset = pd.concat([dataset, atoms_df], axis=1)
    train_idx, test_idx = get_split(dataset, 'elements', element_to_omit_from_training_data)
    
    print('num train samples: '+ str(len(train_idx)))
    print('num test samples: '+ str(len(test_idx)))

    train_data = [d[idx] for idx in train_idx.to_list()]
    test_data = [d[idx] for idx in test_idx.to_list()]

    # train_dataloader = get_data_loader(train_data, target, batch_size=batch_size)
    # test_dataloader = get_data_loader(test_data, target, batch_size=batch_size)
    
    # return train_dataloader, test_dataloader, 
    return train_data, test_data


def sample_list_to_dataloader(sample_list, target, batch_sz):
    return get_data_loader(sample_list, target, batch_size=batch_sz)


def load_zscore_sample(results_df, sample_list, n_samples, target, z_range):

    assert z_range in ['high', 'low'], 'z_range not defined'

    if z_range == 'high':
        subset_df = results_df.nlargest(n_samples, 'z_score_err')
    elif z_range == 'low':
        subset_df = results_df.nsmallest(n_samples, 'z_score_err')
    
    subset_df_idx = subset_df.index.values.tolist()
    subset_list = [sample_list[i] for i in subset_df_idx]
    subset_dataloader = get_data_loader(subset_list, target)

    return subset_dataloader


def get_split(df, group_label,group_value):
    if group_label in ['space_group_number','point_group','crystal_system',]:
        index_train = df[df[group_label].astype(str)!=group_value].index
        index_test = df[df[group_label].astype(str)==group_value].index
    elif group_label in ['elements','period','group',]:
        index_train = df[df[group_label].apply(lambda x: group_value not in x)].index
        index_test = df[df[group_label].apply(lambda x: group_value in x)].index
    elif group_label == 'greater_than_nelements':
        index_train = df[df['nelements']<=int(group_value)].index
        index_test = df[df['nelements']>int(group_value)].index        
    else:
        raise NotImplementedError
    return index_train, index_test

def get_data_loader(atoms_array, target, batch_size=1, workers=0):
    from torch.utils.data import DataLoader

    neighbor_strategy="k-nearest"
    atom_features="cgcnn"
    use_canonize=True
    line_graph=True
    pin_memory=False

    mem = []
    for i, ii in enumerate(atoms_array):
        info = {}
        info["atoms"] = ii['atoms']
        info["prop"] = ii[target]
        info["jid"] = str(i)
        mem.append(info)

    test_data = get_torch_dataset(
        dataset=mem,
        target="prop",
        neighbor_strategy=neighbor_strategy,
        atom_features=atom_features,
        use_canonize=use_canonize,
        line_graph=line_graph,
    )

    collate_fn = test_data.collate_line_graph

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    return test_loader

# def force_wts_into_model(new_model_wts, empty_model, og_layer_names, old_model_state_dict):

#     new_model_wt_dict = copy.deepcopy(old_model_state_dict)

#     for layer, new_param in zip(og_layer_names, new_model_wts):
#         if new_param.shape == old_model_state_dict[layer].shape:
#             new_model_wt_dict[layer] = new_param
#         else:
#             print(layer+" incompatible")

#     err_layers = empty_model.load_state_dict(new_model_wt_dict, strict=False)
#     print(err_layers)

#     return empty_model

def force_wts_into_model(og_layer_names, new_model_wts, empty_model, old_model_state_dict):

    new_model_wt_dict = copy.deepcopy(old_model_state_dict)

    for layer, new_param in zip(og_layer_names, new_model_wts):
        if new_param.shape == old_model_state_dict[layer].shape:
            new_model_wt_dict[layer] = new_param
        else:
            print(layer+" incompatible")

    err_layers = empty_model.load_state_dict(new_model_wt_dict, strict=False)
    print(err_layers)

    return empty_model

def move_ll_to_cpu(loss_landscape):

    loss_landscape_off_gpu = []

    for row in loss_landscape:
        tmp_row = []
        for itm in row:
            tmp_row.append(itm.detach().cpu().numpy())
        loss_landscape_off_gpu.append(tmp_row)
    return loss_landscape_off_gpu
