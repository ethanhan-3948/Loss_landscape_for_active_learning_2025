from src import botcher_hessian as hess
from src import botcher_utilities as util
import numpy as np
import torch
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from collections import OrderedDict
from torchinfo import summary
import pandas as pd
from util.utils_AD import *

def npvec_to_tensorlist(vec, params):
    """ Convert a numpy vector to a list of tensor with the same dimensions as params

        Args:
            vec: a 1D numpy vector
            params: a list of parameters from net

        Returns:
            rval: a list of tensors with the same shape as params
    """
    loc = 0
    rval = []
    #print(f'Converting vector of size {vec.size} to tensor list with {len(params)} parameters')
    for p in params:
        numel = p.data.numel()
        rval.append(torch.from_numpy(vec[loc:loc+numel]).view(p.data.shape).float())
        loc += numel
    #print(f'The vector has a {loc} elements and the net has {vec.size} parameters')
    assert loc == vec.size, f'ERROR: The vector has a {loc} elements and the net has {vec.size} parameters'
    return rval 

def gradtensor_to_npvec(net, all_params=True):
    """ Extract gradients from net, and return a concatenated numpy vector.

        Args:
            net: trained model
            all_params: If all_params, then gradients w.r.t. BN parameters and bias
            values are also included. Otherwise only gradients with dim > 1 are considered.

        Returns:
            a concatenated numpy vector containing all gradients
    """
    filter = lambda p: all_params or len(p.data.size()) >= 1

    # tmp_list = [p.grad.data.cpu().numpy().ravel() for p in net.parameters() if filter(p)]
    tmp_list = []
    for p in net.parameters():
        if p.grad is None:
            tmp_list.append(np.zeros(p.data.numel(), dtype=np.float32))
        elif filter(p):
            tmp_list.append(p.grad.data.cpu().numpy().ravel())
    # print("grad tensor list size: ", len(tmp_list))

    tmp_list = np.concatenate(tmp_list)

    # print(f'grad tensor list shape: {tmp_list.shape}, dtype: {tmp_list.dtype}')

    return tmp_list

def eval_hess_vec_prod(vec, params, net, loss_func, inputs_dataloader, use_cuda=False):
    from torch.autograd import Variable
    """
    Evaluate product of the Hessian of the loss function with a direction vector "vec".
    The product result is saved in the grad of net.

    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net (ignoring biases and BN parameters).
        net: model with trained parameters.
        criterion: loss function.
        inputs: nn inputs.
        outputs: desired nn outputs.
        use_cuda: use GPU.
    """

    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]
    
    device = next(net.parameters()).device
    
 
    #print(f"Using device: {device}")
    net.eval()
    net.zero_grad() # clears grad for every parameter in the net
    
    loss = torch.tensor(0.0).to(device)
    for _, i in enumerate(inputs_dataloader):
        pred_outputs = net((i[0].to(device), i[1].to(device)))
        outputs = i[2].to(device)
        loss = torch.add(loss, loss_func(pred_outputs,outputs))
    
    loss = loss/len(inputs_dataloader)
    grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True, allow_unused=True)

    # Compute inner product of gradient with the direction vector
    prod = Variable(torch.zeros(1)).type(type(grad_f[0].data))

    tmp = []
    for i in range(len(vec)):
        if (grad_f[i] is not None) and (vec[i] is not None):
            tmp.append((grad_f[i].to(device) * vec[i].to(device)).cpu().sum())
    prod =+ sum(tmp)
    # prod += sum([(grad_f[i] * vec[i]).cpu().sum() for i in range(len(vec))])

    # Compute the Hessian-vector product, H*v
    # prod.backward() computes dprod/dparams for every parameter in params and
    # accumulate the gradients into the params.grad attributes
    prod.backward()

def min_max_hessian_eigs(net, data_loader, loss_func, rank=0, use_cuda=False, verbose=False, all_params=True):
    from scipy.sparse.linalg import LinearOperator, eigsh
    import time

    """
        Compute the largest and the smallest eigenvalues of the Hessian marix.

        Args:
            net: the trained model.
            inputs: nn inputs.
            outputs: desired nn outputs.
            criterion: loss function.
            rank: rank of the working node.
            use_cuda: use GPU
            verbose: print more information
            all_params: use all nn parameters

        Returns:
            maxeig: max eigenvalue
            mineig: min eigenvalue
            hess_vec_prod.count: number of iterations for calculating max and min eigenvalues
    """
    
    if all_params:
        params = [p for p in net.parameters()]
    else:
        params = [p for p in net.parameters() if len(p.size()) >= 1]
        
    N = sum(p.numel() for p in params)
    print(f'Total number of parameters: {N}')
    print(f'len params: {len(params)}')

    def hess_vec_prod(vec):
        hess_vec_prod.count += 1  # simulates a static variable
        vec = npvec_to_tensorlist(vec, params)
        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, loss_func, data_loader, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0: print("Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        grad_tensor = gradtensor_to_npvec(net,all_params)
        #print(f'grad_tensor: {grad_tensor.shape}')
        return grad_tensor
        
    hess_vec_prod.count = 0
    if verbose and rank == 0: print("Rank %d: computing max eigenvalue" % rank)

    A = LinearOperator((N, N), matvec=hess_vec_prod)
  
    eigvals, eigvecs = eigsh(A, k=2, which='LM', tol=1e-2)
    maxeig = eigvals[0]
    maxeigvec = eigvecs[:, 0]

    second_eig = eigvals[1]
    second_eigvec = eigvecs[:, 1]

    if second_eig < maxeig:
        maxeig, second_eig = second_eig, maxeig
        maxeigvec, second_eigvec = second_eigvec, maxeigvec

    if verbose and rank == 0: print('max eigenvalue = %f' % maxeig)

    # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
    # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
    shift = maxeig*1.0
    def shifted_hess_vec_prod(vec):
        # print(f'shifted_hess_vec_prod: {vec.shape}')
        # print(f'params: {len(params)}')
        return hess_vec_prod(vec) - shift*vec

    if verbose and rank == 0: print("Rank %d: Computing shifted eigenvalue" % rank)

    A = LinearOperator((N, N), matvec=shifted_hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=1, which='LM', tol=1e-2)
    eigvals = eigvals + shift
    mineig = eigvals[0]
    mineigvec = eigvecs

    if verbose and rank == 0: print('min eigenvalue = ' + str(mineig))

    if maxeig <= 0 and mineig > 0:
        maxeig, mineig = mineig, maxeig
        maxeigvec, mineigvec = mineigvec, maxeigvec

    return maxeig, mineig, maxeigvec, mineigvec, second_eig, second_eigvec


def force_wts_into_model(og_layer_names, new_model_wts, empty_model, old_model_state_dict):
    from copy import deepcopy
    new_model_wt_dict = deepcopy(old_model_state_dict)

    for layer, new_param in zip(og_layer_names, new_model_wts):
        if new_param.shape == old_model_state_dict[layer].shape:
            new_model_wt_dict[layer] = new_param
        else:
            print(layer+" incompatible")

    err_layers = empty_model.load_state_dict(new_model_wt_dict, strict=False)
    print(err_layers)

    return empty_model