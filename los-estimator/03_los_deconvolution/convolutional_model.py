import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt

def los_distro_converter(los):
    """Converts from distribution of discharge to distribution of presence in icu (monotonically decreasing)."""
    if len(los.shape) ==1:
        los2 = 1-np.cumsum(los)
    else:
        los2 = 1 - np.cumsum(los, axis=1)
    return los2

def calc_its_convolution(inc,los_distro1, transition_rate,delay,los_cutoff,gaussian_spread=False):
    if not gaussian_spread:
        return calc_its_convolution_old_and_gold(inc,los_distro1, transition_rate,delay,los_cutoff)
    los_distro = los_distro_converter(los_distro1)
    gaussian = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
    inc2 = transition_rate * np.concatenate([np.zeros(int(delay)),inc])    
    admissions = convolve(inc2, gaussian, mode="full")
    its = convolve(admissions, los_distro, mode="full")
    its[:los_cutoff] = 0
    its = its[:len(inc)]
    return its

from numba import njit
import numpy as np


@njit
def convolve_variable_kernel(admissions, los_distro):    
    adm_len = admissions.shape[0]
    n_kernel, t_kernel = los_distro.shape
    result = np.zeros(adm_len)
    for t in range(adm_len):
        for kernel_pos in range(t_kernel):
            if t + kernel_pos < adm_len:
                result[t + kernel_pos] += admissions[t] * los_distro[min(t,n_kernel-1), kernel_pos]
    return result

import matplotlib.pyplot as plt
def calc_its_convolution_old_and_gold(inc,los_distro1, transition_rate,delay,los_cutoff):
    if len(los_distro1.shape) == 1:
        los_distro1  = np.tile(los_distro1, (len(inc), 1))
        
    los_distro = los_distro_converter(los_distro1)

    admissions = inc * transition_rate
    delay_zeros = np.zeros([los_distro.shape[0],int(delay)])
    los_distro = np.concatenate([delay_zeros,los_distro],axis=-1)
    
    its = convolve_variable_kernel(admissions,los_distro)

    
    
    # # Smooth transition between days for floating point optimization
    # its = np.concatenate([np.zeros(int(delay) + 1), its,[ 0]])
    # intraday_delay = delay - int(delay)
    # its = its[1:] * (1 - intraday_delay) + its[:-1] * intraday_delay

    # Remove beginning and end of the signal according to the los_cutoff (los_cutoff is the point, where the main mass of the los_distro is)
    its[:los_cutoff] = 0

    return its

def calc_its_convolution_original(inc,los_distro1, transition_rate,delay,los_cutoff):
    los_distro = los_distro_converter(los_distro1)
    
    admissions = inc * transition_rate
    los_distro = np.concatenate([np.zeros(int(delay)),los_distro])
    its = convolve(admissions, los_distro, mode="full")
    
    # # Smooth transition between days for floating point optimization
    # its = np.concatenate([np.zeros(int(delay) + 1), its,[ 0]])
    # intraday_delay = delay - int(delay)
    # its = its[1:] * (1 - intraday_delay) + its[:-1] * intraday_delay

    # Remove beginning and end of the signal according to the los_cutoff (los_cutoff is the point, where the main mass of the los_distro is)
    its[:los_cutoff] = 0
    its = its[:-len(los_distro) + 1]

    return its


def mse(pred,real):
    pred = pred[:len(real)]
    return np.mean((pred - real)**2)

def objective_function_conv(params,inc,icu,daily_los,los_cutoff):
    transition_rate,delay  = params

    pred = calc_its_convolution(inc,daily_los,transition_rate,delay,los_cutoff)
    return mse(pred,icu)
