import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, weibull_min, norm, expon, gamma, beta, cauchy, t, invgauss
from convolutional_model import calc_its_convolution
import matplotlib.pyplot as plt
from compartmental_model import objective_function_compartmental,calc_its_comp
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

sentinel_los_berlin = np.array([0.01387985, 0.04901323, 0.0516157 , 0.05530254, 0.04706137,
       0.05421817, 0.05074821, 0.04576014, 0.03838647, 0.03318152,
       0.03513338, 0.02819345, 0.03079592, 0.02645847, 0.02884407,
       0.02775971, 0.01886792, 0.01474734, 0.01800043, 0.01583171,
       0.01778356, 0.01778356, 0.00975927, 0.01257862, 0.01236174,
       0.01322923, 0.01040989, 0.0095424 , 0.00910865, 0.0095424 ,
       0.00845804, 0.00889178, 0.0071568 , 0.00910865, 0.01236174,
       0.00650618, 0.00563869, 0.00693993, 0.00780742, 0.00585556,
       0.00542182, 0.00498807, 0.00542182, 0.00585556, 0.00216873,
       0.00281935, 0.00672305, 0.00498807, 0.00368684, 0.00195185,
       0.00130124, 0.00346996, 0.00303622, 0.00195185, 0.00412058,
       0.0023856 , 0.00195185, 0.0023856 , 0.00130124, 0.00195185,
       0.00021687, 0.00216873, 0.00043375, 0.00108436, 0.00043375,
       0.00346996, 0.00173498, 0.00021687, 0.00043375, 0.00151811,
       0.00173498, 0.00195185, 0.00130124, 0.00173498, 0.00151811,
       0.00260247, 0.00065062, 0.00151811, 0.        , 0.00043375,
       0.00086749, 0.00065062, 0.00086749, 0.00108436, 0.00151811,
       0.00043375, 0.00130124, 0.00151811, 0.        , 0.00086749,
       0.00173498, 0.00130124, 0.        , 0.        , 0.00108436,
       0.        , 0.00043375, 0.00043375, 0.        , 0.00086749,
       0.00043375, 0.00086749, 0.        , 0.        , 0.        ,
       0.00021687, 0.        , 0.        , 0.        , 0.00065062,
       0.        , 0.00021687, 0.00043375, 0.00130124, 0.        ,
       0.00021687, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.00065062, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.00108436,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.00086749, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.00043375, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        ])


distributions = {
    "lognorm": [1, 0],
    "weibull": [1, 15],
    "gaussian": [0, 1],
    "exponential": [1],
    "gamma": [2, 2],
    "beta": [2, 2],
    "cauchy": [0, 1],
    "t": [10, 0, 1],
    "invgauss": [1, 0],
    "linear": [40],
    "block": [],
    "sentinel": [],
}

boundaries = {
    "lognorm": [(0, None), (0, None)],
    "weibull": [(1, None), (0, None)],
    "gaussian": [(0, None), (0, None)],
    "exponential": [(0.001, None)],
    "gamma": [(0, None), (0, None)],
    "beta": [(0, None), (0, None)],
    "cauchy": [(0, None), (0, None)],
    "t": [(0, None), (0, None), (0, None)],
    "invgauss": [(0, None), (0, None)],
    "linear": [(0, None)],
    "block": [],
    "sentinel": [],
}

DISTROS = {
    "lognorm":   lambda x, sigma, μ: lognorm.pdf(x, s=sigma,   scale=np.exp(μ)),
    "weibull":   lambda x, k, λ: weibull_min.pdf(x, c=k, scale=λ),
    "gaussian":  lambda x, μ, sigma: norm.pdf(x, loc=μ,     scale=sigma),
    "exponential":lambda x, λ:  expon.pdf(x, scale=1/λ),
    "gamma":     lambda x, a, s: gamma.pdf(x, a=a,      scale=s),
    "beta":      lambda x, a, b: beta.pdf(x, a=a,       b=b),
    "cauchy":    lambda x, μ, s: cauchy.pdf(x, loc=μ,   scale=s),
    "t":         lambda x, v, μ, s: t.pdf(x, df=v,      loc=μ, scale=s),
    "invgauss":  lambda x, μ, loc: invgauss.pdf(x, μ, loc=loc),
    "linear":    lambda x, L: np.clip(-x/L + 1, 0, None),
    "block":     lambda x: np.eye(1, len(x), 1, dtype=float).ravel(),
    "sentinel":  lambda x: np.asarray(sentinel_los_berlin, dtype=float),
}

def generate_kernel(distro, fun_params, kernel_size):
    *params, scaling_fac = fun_params
    pdf = DISTROS[distro]
    x = np.arange(kernel_size, dtype=float) * scaling_fac
    kernel = pdf(x, *params)
    result = kernel / kernel.sum()
    return result


def get_error_fun(error_fun):
    if error_fun == "mse":
        return mse
    elif error_fun == "weighted_mse":
        return weighted_mse
    else:
        raise ValueError(f"Unknown Error Function: {error_fun}")



# Objective function for direct kernel fit
def objective_fit_kernel_to_sentinel(distro):
    def objective_function(params, observed):
        kernel = generate_kernel(distro, params, observed.shape[0])
        return mse(kernel, observed)
    return objective_function

def stitch_together_multidim_kernel(past_kernels, kernel):
    kernels = np.zeros((len(past_kernels)+1,kernel.shape[0]))
    kernels[:len(past_kernels)] = past_kernels
    kernels[-1] = kernel
    return kernels


# Objective function for direct kernel fit
def objective_fit_kernel_to_series(distro, kernel_width, los_cutoff, error_fun=mse):
    def objective_function(params, inc, icu, past_kernels=None):
        transition_rate, delay, *fun_params = params

        kernel = generate_kernel(distro, fun_params, kernel_width)
        if past_kernels is not None:
            kernel = stitch_together_multidim_kernel(past_kernels, kernel)

        observed = calc_its_convolution(inc, kernel, transition_rate, delay, los_cutoff)
        res = error_fun(icu[los_cutoff:], observed[los_cutoff:])
        return res
    return objective_function



def fit_kernel_to_series(
        distro,
        x_train,
        y_train,
        x_test,
        y_test,
        kernel_width,
        los_cutoff,
        curve_init_params,
        curve_fit_boundaries,
        distro_boundaries=None,
        distro_init_params=None,
        past_kernels=None,
        method="L-BFGS-B",
        error_fun="mse"
    ):
        error_fun = get_error_fun(error_fun)

        if distro_boundaries is None:
            distro_boundaries = boundaries[distro] + [(None, None)]

        if not distro_init_params:
            stretching_init = 1
            distro_init_params = distributions[distro] + [stretching_init]

        params = np.concatenate((curve_init_params, distro_init_params))

        obj_fun = objective_fit_kernel_to_series(distro, kernel_width, los_cutoff, error_fun)

        minimize_args = (x_train, y_train)
        if past_kernels is not None:
            minimize_args = (x_train, y_train, past_kernels)

        result = minimize(
            obj_fun,
            params,
            args=minimize_args,
            bounds=curve_fit_boundaries + distro_boundaries,
            method=method,
        )

        # Generate the fitted kernel and predictions
        fitted_kernel = generate_kernel(distro, result.x[2:], kernel_width)        
        if past_kernels is None:            
            y_pred = calc_its_convolution(x_test, fitted_kernel, *result.x[:2], los_cutoff)
        else:
            kernels = stitch_together_multidim_kernel(past_kernels, fitted_kernel)
            y_pred = calc_its_convolution(x_test, kernels, *result.x[:2], los_cutoff)
        
        train_len = len(x_train)
        train_err = obj_fun(result.x, x_train, y_train, past_kernels)
        test_err = obj_fun(result.x, x_test[train_len-los_cutoff:], y_test[train_len-los_cutoff:], past_kernels)

        relative_error = np.abs((y_pred - y_test) / (y_test+1))
        relative_error[np.isnan(relative_error)] = 0

        return {
            "params": result.x,
            "kernel": fitted_kernel,
            "curve": y_pred,
            "train_error": train_err,
            "test_error": test_err,
            "minimization_result": result,
            "relative_error": relative_error,
        }

def objective_compartemental(error_fun):    
    def objective_function(params,inc,icu,los_cutoff):
        discharge_rate, transition_rate,delay  = params
        pred = calc_its_comp(inc,discharge_rate,transition_rate,delay,init=icu[0])
        return error_fun(pred[los_cutoff:len(icu)],icu[los_cutoff:])
    return objective_function


def fit_SEIR(x_train, y_train,x_test,y_test, initial_guess_comp,los_cutoff,method="TNC"):
    error_fun = get_error_fun("mse")
    obj_fun = objective_compartemental(error_fun)

    result = minimize(
            obj_fun,
            initial_guess_comp,
            args=(x_train,y_train,los_cutoff),
            method=method,
            bounds=[(0, 1),(1, 1),(0,0)]
        )
    y_pred = calc_its_comp(x_test, *result.x, y_test[0])

    test_x = x_test[len(x_train)-los_cutoff:]
    test_y = y_test[len(x_train)-los_cutoff:]


    train_err = obj_fun(result.x, x_train, y_train, los_cutoff)
    test_err = obj_fun(result.x, test_x, test_y, los_cutoff)
    result_dict = {
        "params": result.x,
        "kernel": np.zeros(1),
        "curve": y_pred,
        "train_error": train_err,
        "test_error": test_err,
        "minimization_result": result,
    }
        
    return result_dict
