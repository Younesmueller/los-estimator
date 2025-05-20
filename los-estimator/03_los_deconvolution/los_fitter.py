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




# Objective function for direct kernel fit
def gen_obj_fun(distro):
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
def objective_direct_kernel_fit_new(distro, kernel_width, los_cutoff, error_fun=mse):
    def objective_function(params, inc, icu, past_kernels=None):
        transition_rate, delay, *fun_params = params

        kernel = generate_kernel(distro, fun_params, kernel_width)
        if past_kernels is not None:
            kernel = stitch_together_multidim_kernel(past_kernels, kernel)

        observed = calc_its_convolution(inc, kernel, transition_rate, delay, los_cutoff)
        res = error_fun(icu[los_cutoff:], observed[los_cutoff:])
        return res
    return objective_function

def fit_kernel_distro_to_data_with_previous_results(
    distro,
    x_train,
    y_train,
    x_test,
    y_test,
    past_kernels, # Shape: x_train.shape[0] - los_cutoff X kernel_width
    kernel_width,
    los_cutoff,
    curve_init_params,
    curve_fit_boundaries,
    distro_boundaries=None,
    distro_init_params=None,
    method="L-BFGS-B",
    error_fun="mse"
):
    if error_fun == "mse":
        error_fun = mse
    elif error_fun == "weighted_mse":
        error_fun = weighted_mse
    else:
        raise Exception("Unknown Error Function")
    if distro_boundaries is None:
        distro_boundaries = boundaries[distro] + [(None,None)]
    if distro_init_params is None or len(distro_init_params) == 0:
        stretching_init = 1
        distro_init_params = distributions[distro] + [stretching_init]
    params = np.concatenate((curve_init_params, distro_init_params))
    obj_fun = objective_direct_kernel_fit_new(distro, kernel_width, los_cutoff, error_fun)
    result = minimize(
        obj_fun,
        params,
        args=(
            x_train,
            y_train,
            past_kernels,
        ),
        bounds=curve_fit_boundaries + distro_boundaries,
        method=method,
    )
    # Generate and plot the fitted kernel
    fitted_kernel = generate_kernel(distro, result.x[2:], kernel_width)
    train_err = obj_fun(result.x, x_train, y_train, past_kernels)
    kernels = stitch_together_multidim_kernel(past_kernels, fitted_kernel)

    y_pred = calc_its_convolution(x_test, kernels, *result.x[:2], los_cutoff)
    
    
    # plt.plot(y_test);plt.plot(y_pred); plt.show()

    train_length = len(x_train)
    test_err = obj_fun(result.x,
                       x_test[train_length-los_cutoff:],
                       y_test[train_length-los_cutoff:],
                       #TODO: This cannot work!!
                       past_kernels)
    
    result_dict = {}
    result_dict["params"] = result.x
    result_dict["kernel"] = fitted_kernel
    result_dict["curve"] = y_pred
    result_dict["train_error"] = train_err
    result_dict["test_error"] = test_err
    result_dict["minimization_result"] = result
    return result_dict



def fit_kernel_distro_to_data_new(
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
        if error_fun == "mse":
            error_fun = mse
        elif error_fun == "weighted_mse":
            error_fun = weighted_mse
        else:
            raise ValueError(f"Unknown Error Function: {error_fun}")

        if distro_boundaries is None:
            distro_boundaries = boundaries[distro] + [(None, None)]
        if not distro_init_params:
            stretching_init = 1
            distro_init_params = distributions[distro] + [stretching_init]

        params = np.concatenate((curve_init_params, distro_init_params))

        obj_fun = objective_direct_kernel_fit_new(distro, kernel_width, los_cutoff, error_fun)

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


def fit_SEIR(x_train, y_train,x_test,y_test, initial_guess_comp,los_cutoff):
    result = minimize(
            objective_function_compartmental,
            initial_guess_comp,
            args=(x_train,y_train,los_cutoff),
            method="L-BFGS-B",
            bounds=[(0, 1),(1, 1),(0,0)]
        )
    y_pred = calc_its_comp(x_test, *result.x,y_test[0])

    result_dict = {}
    result_dict["params"] = result.x
    result_dict["kernel"] = np.zeros(1)
    result_dict["curve"] = y_pred
    result_dict["train_error"] = [np.nan]
    result_dict["test_error"] = [np.nan]
    result_dict["minimization_result"] = result
        
    return result_dict



########################################################################################
########################################################################################
########################################################################################
########################################################################################

# Objective function for direct kernel fit
def objective_direct_kernel_fit(distro, kernel_length, los_cutoff,error_fun=mse):
    def objective_function(params, inc, icu):
        transition_rate, delay, *fun_params = params
        kernel = generate_kernel(distro, fun_params, kernel_length)
        observed = calc_its_convolution(inc, kernel, transition_rate, delay, los_cutoff)
        return error_fun(icu[los_cutoff:], observed[los_cutoff:])
    return objective_function

# Objective function for direct kernel fit
def objective_direct_kernel_fit_with_previous_results(distro, kernel_length, los_cutoff, error_fun=mse):
    def objective_function(params, inc, icu, past_kernels):
        transition_rate, delay, *fun_params = params
        kernel = generate_kernel(distro, fun_params, kernel_length)
        kernels = np.zeros((len(past_kernels)+1,kernel_length))
        kernels[:-1] = past_kernels
        kernels[-1] = kernel
        observed = calc_its_convolution(inc, kernels, transition_rate, delay, los_cutoff)
        return error_fun(icu[los_cutoff:], observed[los_cutoff:])
    return objective_function

def test_objective_direct_kernel_fit_new_equivalence():
    # Generate synthetic data
    np.random.seed(42)
    kernel_length = 10
    los_cutoff = 2
    inc = np.random.rand(20)
    icu = np.random.rand(20)
    distro = "lognorm"
    params = [0.5, 1.0, 1.0, 0.0, 1.0]  # transition_rate, delay, sigma, mu, scaling_fac

    # No past_kernels case
    obj_new = objective_direct_kernel_fit_new(distro, kernel_length, los_cutoff)
    obj_old = objective_direct_kernel_fit(distro, kernel_length, los_cutoff)
    val_new = obj_new(params, inc, icu)
    val_old = obj_old(params, inc, icu)
    assert np.equal(val_new, val_old), f"Mismatch without past_kernels: {val_new} vs {val_old}"
    

    # With past_kernels case
    past_kernels = [np.ones(kernel_length) / kernel_length]
    obj_new = objective_direct_kernel_fit_new(distro, kernel_length, los_cutoff)
    obj_old_prev = objective_direct_kernel_fit_with_previous_results(distro, kernel_length, los_cutoff)
    val_new = obj_new(params, inc, icu, past_kernels=past_kernels)
    val_old_prev = obj_old_prev(params, inc, icu, past_kernels)
    assert np.equal(val_new, val_old_prev), f"Mismatch with past_kernels: {val_new} vs {val_old_prev}"

    print("objective_direct_kernel_fit_new behaves identically to the old functions.")

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################

def fit_kernel_distro_to_data(
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
    method="L-BFGS-B",
    error_fun="mse"
):
    if error_fun == "mse":
        error_fun = mse
    elif error_fun == "weighted_mse":
        error_fun = weighted_mse
    else:
        raise ValueError(f"Unknown Error Function: {error_fun}")

    if distro_boundaries is None:
        distro_boundaries = boundaries[distro] + [(None, None)]
    if not distro_init_params:
        stretching_init = 1
        distro_init_params = distributions[distro] + [stretching_init]

    params = np.concatenate((curve_init_params, distro_init_params))

    obj_fun = objective_direct_kernel_fit_new(distro, kernel_width, los_cutoff, error_fun)

    result = minimize(
        obj_fun,
        params,
        args=(x_train, y_train),
        bounds=curve_fit_boundaries + distro_boundaries,
        method=method,
    )

    # Generate the fitted kernel and predictions
    fitted_kernel = generate_kernel(distro, result.x[2:], kernel_width)
    train_err = obj_fun(result.x, x_train, y_train)
    y_pred = calc_its_convolution(x_test, fitted_kernel, *result.x[:2], los_cutoff)

    # Calculate test error (ensure correct slicing)
    train_length = len(x_train)
    test_x = x_test[train_length-los_cutoff:]
    test_y = y_test[train_length-los_cutoff:]
    test_err = obj_fun(result.x, test_x, test_y)

    # Collect results
    return {
        "params": result.x,
        "kernel": fitted_kernel,
        "curve": y_pred,
        "train_error": train_err,
        "test_error": test_err,
        "minimization_result": result,
    }

def test_fit_kernel_distro_to_data_new_equivalence():
    # Todo ensure that
    # train legnth is longer than kernel width 
    # that los_cutoff is smaller than train length
    # ...    

    # Generate synthetic data
    np.random.seed(123)
    kernel_width = 120
    los_cutoff = kernel_width
    n_train = 140
    n_test = 160
    np.random.seed(42)
    x_test = np.random.rand(n_test)
    y_test = np.random.rand(n_test)
    x_train = x_test[:n_train]
    y_train = y_test[:n_train]
    
    distro = "lognorm"
    curve_init_params = [0.5, 1.0]  # transition_rate, delay
    curve_fit_boundaries = [(0, 1), (0, 2)]
    distro_init_params = None
    distro_boundaries = None

    # No past_kernels case    
    res_old = fit_kernel_distro_to_data(
        distro, x_train, y_train, x_test, y_test,
        kernel_width, los_cutoff,
        curve_init_params, curve_fit_boundaries,
        distro_boundaries, distro_init_params,
        method="L-BFGS-B",
        error_fun="mse"
    )
    res_new = fit_kernel_distro_to_data_new(
        distro, x_train, y_train, x_test, y_test,
        kernel_width, los_cutoff,
        curve_init_params, curve_fit_boundaries,
        distro_boundaries, distro_init_params,
        past_kernels=None,
        method="L-BFGS-B",
        error_fun="mse"
    )

    # Compare main outputs
    assert np.allclose(res_new["params"], res_old["params"]), f"Params mismatch: {res_new['params']} vs {res_old['params']}"
    assert np.allclose(res_new["kernel"], res_old["kernel"]), f"Kernel mismatch"
    assert np.allclose(res_new["curve"], res_old["curve"]), f"Curve mismatch"
    assert np.allclose(res_new["train_error"], res_old["train_error"]), f"Train error mismatch"
    assert np.allclose(res_new["test_error"], res_old["test_error"]), f"Test error mismatch"

    # With past_kernels case
    past_kernels = [np.ones(kernel_width) / kernel_width]
    res_new_past = fit_kernel_distro_to_data_new(
        distro, x_train, y_train, x_test, y_test,
        kernel_width, los_cutoff,
        curve_init_params, curve_fit_boundaries,
        distro_boundaries, distro_init_params,
        past_kernels=past_kernels,
        method="L-BFGS-B",
        error_fun="mse"
    )
    res_old_past = fit_kernel_distro_to_data_with_previous_results(
        distro, x_train, y_train, x_test, y_test,
        past_kernels,
        kernel_width, los_cutoff,
        curve_init_params, curve_fit_boundaries,
        distro_boundaries, distro_init_params,
        method="L-BFGS-B",
        error_fun="mse"
    )

    assert np.allclose(res_new_past["params"], res_old_past["params"]), f"Params mismatch (past): {res_new_past['params']} vs {res_old_past['params']}"
    assert np.allclose(res_new_past["kernel"], res_old_past["kernel"]), f"Kernel mismatch (past)"
    assert np.allclose(res_new_past["curve"], res_old_past["curve"]), f"Curve mismatch (past)"
    assert np.allclose(res_new_past["train_error"], res_old_past["train_error"]), f"Train error mismatch (past)"    
    assert np.allclose(res_new_past["test_error"], res_old_past["test_error"]), f"Test error mismatch (past): {res_new_past['test_error']-res_old_past['test_error']}"

    print("fit_kernel_distro_to_data_new is consistent with the old functions.")
