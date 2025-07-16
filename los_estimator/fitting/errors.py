import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def weighted_mse(x,y):
    le = len(x)
    weights = np.exp(np.linspace(0,2,le))
    weights /= weights.sum()
    return np.sum(((x - y) ** 2)*weights)

@njit
def mse(x, y):
    return np.mean((x - y) ** 2)



def get_error_fun(error_fun):
    if error_fun == "mse":
        return mse
    elif error_fun == "weighted_mse":
        return weighted_mse
    else:
        raise ValueError(f"Unknown Error Function: {error_fun}")
