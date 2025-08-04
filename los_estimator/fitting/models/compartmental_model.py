import numpy as np
from numba import njit



@njit
def calc_its_comp(inc, discharge_rate, transition_rate, delay,init):
    int_delay = int(delay)
    beds = inc * transition_rate
    beds = update_beds(beds, init, (1-discharge_rate))
    intraday_delay = delay-int(delay)
    
    beds_ext = np.zeros(beds.shape[0] + int_delay + 2, dtype=beds.dtype)
    beds_ext[int_delay + 1: -1] = beds
    beds_ext[-1] = beds[-1]
    beds = beds_ext

    beds = beds[1:]*(1-intraday_delay)+beds[:-1]*intraday_delay
    beds = beds[:-1]
    return beds


@njit
def update_beds(beds, init, rate):
    beds[0] += init
    for i in range(len(beds)-1):
        beds[i+1] += beds[i] * rate
    return beds

def mse(pred,real):
    pred = pred[:len(real)]
    return np.mean((pred - real)**2)

def objective_function_compartmental(params,inc,icu,los_cutoff):
    discharge_rate,transition_rate,delay  = params

    pred = calc_its_comp(inc,discharge_rate,transition_rate,delay,init=icu[0])

    return mse(pred[los_cutoff:],icu[los_cutoff:])
