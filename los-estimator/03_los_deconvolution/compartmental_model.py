import numpy as np
from numba import njit


def calc_its_comp(inc, discharge_rate, transition_rate, delay,init):
    beds = inc * transition_rate
    beds = update_beds(beds, init, (1-discharge_rate))
    intraday_delay = delay-int(delay)
    beds = np.concatenate([np.zeros(int(delay)+1),beds,[beds[-1]]])
    beds = beds[1:]*(1-intraday_delay)+beds[:-1]*intraday_delay
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
