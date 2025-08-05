import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.optimize import minimize
from los_estimator.fitting.models.convolutional_model import calc_its_convolution
from los_estimator.fitting.models.compartmental_model import calc_its_comp
from .distributions import Distributions
from .fit_results import SingleFitResult
from .errors import ErrorFunctions

def combine_past_kernel(past_kernels, kernel):
    return np.vstack([*past_kernels, kernel])


def get_objective_convolution(distro, kernel_width, los_cutoff, error_fun):
    def objective_function(model_config, inc, icu, past_kernels=None):
        kernel = Distributions.generate_kernel(distro, model_config, kernel_width)
        if past_kernels is not None:
            kernel = combine_past_kernel(past_kernels, kernel)
        
        observed = calc_its_convolution(inc, kernel, los_cutoff)
        res = error_fun(icu[los_cutoff:], observed[los_cutoff:])
        return res
    return objective_function

def initialize_distro_parameters(distro, distro_boundaries, distro_init_params):
    
    if distro_boundaries is None:
        stretch_factor_bounds = [(None, None)]
        distro_boundaries = Distributions[distro].boundaries + stretch_factor_bounds

    if distro_init_params is None or len(distro_init_params) == 0:
        stretching_init = 1
        distro_init_params = Distributions[distro].init_values + [stretching_init]
    
    return distro_boundaries,distro_init_params


def fit_convolution(
        distro,
        train_data,
        test_data,
        kernel_width,
        los_cutoff,
        distro_boundaries=None,
        distro_init_params=None,
        past_kernels=None,
        method="L-BFGS-B",
        error_fun="mse",
    ):
        
        x_train, y_train = train_data
        x_test, y_test = test_data
        error_fun = ErrorFunctions[error_fun]

        distro_boundaries, distro_init_params = initialize_distro_parameters(distro, distro_boundaries, distro_init_params)

        obj_fun = get_objective_convolution(distro, kernel_width, los_cutoff, error_fun)
        
        args = (x_train,y_train,past_kernels,)        

        result = minimize(
            obj_fun,
            x0=distro_init_params,
            args=args,
            bounds=distro_boundaries,
            method=method,
        )

        distro_params = result.x        

        # Generate the fitted kernel and predictions
        fitted_kernel = Distributions.generate_kernel(distro, distro_params, kernel_width)        
        if past_kernels is None:            
            y_pred = calc_its_convolution(x_test, fitted_kernel, los_cutoff)
        else:
            kernels = combine_past_kernel(past_kernels, fitted_kernel)
            y_pred = calc_its_convolution(x_test, kernels, los_cutoff)
        
        train_len = len(x_train)
        train_err = obj_fun(result.x, x_train, y_train, *args[2:] )
        test_err = obj_fun(result.x, x_test[train_len-los_cutoff:], y_test[train_len-los_cutoff:],*args[2:])


        fit_results = SingleFitResult(
            distro=distro,
            train_data=x_train,
            test_data=x_test,
            success=result.success,
            minimization_result=result,
            train_error=train_err,
            test_error=test_err,
            kernel=fitted_kernel,
            curve=y_pred,
            model_config=distro_params,
        )

        return fit_results



def objective_compartemental(error_fun):    
    def objective_function(model_config,inc,icu,los_cutoff):
        discharge_rate, transition_rate,delay  = model_config
        pred = calc_its_comp(inc,discharge_rate,transition_rate,delay,init=icu[0])
        return error_fun(pred[los_cutoff:len(icu)],icu[los_cutoff:])
    return objective_function




def fit_compartmental(
        train_data,
        test_data,
        initial_guess_comp,
        los_cutoff,
        method="TNC",
        error_fun="mse",
    ):
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    error_fun = ErrorFunctions[error_fun]
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
    relative_error = np.abs((y_pred - y_test) / (y_test+1))
    relative_error[np.isnan(relative_error)] = 0

    result_obj = SingleFitResult(
        distro="compartmental",
        train_data=x_train,
        test_data=x_test,
        success=result.success,
        minimization_result=result,
        train_error=train_err,
        test_error=test_err,
        rel_train_error=relative_error[:len(x_train)],
        rel_test_error=relative_error[len(x_train):],
        kernel=np.zeros(1),
        curve=y_pred,
        model_config=result.x
    )
        
    return result_obj
