from numba import njit
import numpy as np
import matplotlib.pyplot as plt

def los_distro_converter(los):
    """Converts from distribution of discharge to distribution of presence in icu (monotonically decreasing)."""
    if len(los.shape) == 1:
        raise Exception("los_distro must be 2D")
    los2 = 1 - np.cumsum(los, axis=1)
    return los2

def calc_its_convolution(admissions,los_distro1,los_cutoff):
    if len(los_distro1.shape) == 1:
        los_distro1  = los_distro1[None,:]
    los_distro = los_distro_converter(los_distro1)
    its = convolve_variable_kernel(admissions, los_distro)
    its[:los_cutoff] = 0 # Remove beginning and end of the signal according to the los_cutoff (los_cutoff is the point, where the main mass of the los_distro is)
    return its

@njit
def convolve_variable_kernel(admissions, los_distro):    
    adm_len = admissions.shape[0]
    n_kernel, kernel_len = los_distro.shape
    
    result = np.zeros(adm_len)
    for t in range(adm_len):
        for kernel_pos in range(min(kernel_len, adm_len - t)):
            i_kernel = min(t, n_kernel - 1)
            result[t + kernel_pos] += admissions[t] * los_distro[i_kernel, kernel_pos]
    return result