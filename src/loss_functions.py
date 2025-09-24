import numpy as np

def L2_loss(true, predicted):
    """Calculate the L2 loss (Mean Squared Error)."""
    return ((true - predicted) ** 2).mean()

def L1_loss(true, predicted):
    """Calculate the L1 loss (Mean Absolute Error)."""
    return abs(true - predicted).mean()

def LogCosh_loss(true, predicted):
    """Calculate the Log-Cosh loss."""
    return (np.log(np.cosh(predicted - true))).mean()

def Huber_delta_0_01_loss(true, predicted, delta=0.01):
    """Calculate the Huber loss with delta=0.01."""
    diff = true - predicted
    is_small_error = abs(diff) <= delta
    squared_loss = 0.5 * (diff ** 2)
    linear_loss = delta * (abs(diff) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()