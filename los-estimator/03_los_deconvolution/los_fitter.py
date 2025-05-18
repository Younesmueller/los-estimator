import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, weibull_min, norm, expon, gamma, beta, cauchy, t, invgauss
from convolutional_model import calc_its_convolution
import matplotlib.pyplot as plt
from compartmental_model import objective_function_compartmental,calc_its_comp
def weighted_mse(x,y):
    le = len(x)
    weights = np.exp(np.linspace(0,2,le))
    weights/=weights.sum()
    return np.sum(((x - y) ** 2)*weights)


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
    # "double_linear":[25,.1,75],
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
    # "double_linear": [(0, 200), (0, 1), (0, 200)],
    "block": [],
    "sentinel": [],
}


# Abstract kernel generator function
def generate_kernel(distro, params, kernel_size):
    """Generate a distribution kernel given params and kernel size."""
    *params, scaling_fac = params
    x = np.arange(kernel_size) * scaling_fac
    if distro == "lognorm":
        sigma, mu = params
        kernel = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    elif distro == "weibull":
        k, lambda_ = params
        kernel = weibull_min.pdf(x, c=k, scale=lambda_)
    elif distro == "gaussian":
        mu, sigma = params
        kernel = norm.pdf(x, loc=mu, scale=sigma)
    elif distro == "exponential":
        lambda_ = params[0]
        kernel = expon.pdf(x, scale=1 / lambda_)
    elif distro == "gamma":
        shape, scale = params
        kernel = gamma.pdf(x, a=shape, scale=scale)
    elif distro == "beta":
        a, b = params
        kernel = beta.pdf(x / max(x), a=a, b=b)  # scale x to [0, 1]
    elif distro == "cauchy":
        loc, scale = params
        kernel = cauchy.pdf(x, loc=loc, scale=scale)
    elif distro == "t":
        df, loc, scale = params
        kernel = t.pdf(x, df=df, loc=loc, scale=scale)
    elif distro == "invgauss":
        mu, loc = params
        kernel = invgauss.pdf(x, mu, loc=loc)
    elif distro == "linear":
        kernel = -x/params[0]+1
        kernel[kernel<0] = 0
    # elif distro =="double_linear":
    #     p1x, p1y, p2x = params
    #     x1 = int(p1x+1)
    #     x2 = int(p2x+1)
    #     kernel = np.zeros_like(x)
    #     m = (p1y-1)/p1x
    #     m2 = -p1y/(p2x-p1x)
    #     b2 = p1y - m2*p1x
    #     kernel[:x1] = m*x[:x1] + 1
    #     kernel[x1:x2] = (m2*x + b2)[x1:x2]
    elif distro == "block":
        kernel = np.zeros(kernel_size)
        kernel[1]=1
    elif distro == "sentinel":
        kernel = sentinel_los_berlin
    else:
        raise ValueError(f"Unsupported distribution: {distro}")

    kernel /= kernel.sum()  # Normalize the kernel
    return kernel


# Objective function for direct kernel fit
def gen_obj_fun(distro):
    def objective_function(params, observed):
        kernel = generate_kernel(distro, params, observed.shape[0])
        return mse(kernel, observed)

    return objective_function


# Objective function for direct kernel fit
def objective_direct_kernel_fit(distro, kernel_length, los_cutoff, gaussian_spread=False,error_fun=mse):
    def objective_function(params, inc, icu):
        transition_rate, delay, *fun_params = params
        kernel = generate_kernel(distro, fun_params, kernel_length)
        observed = calc_its_convolution(inc, kernel, transition_rate, delay, los_cutoff,gaussian_spread)
        return error_fun(icu[los_cutoff:], observed[los_cutoff:])

    return objective_function


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
    gaussian_spread=False,
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
    obj_fun = objective_direct_kernel_fit(distro, kernel_width, los_cutoff, gaussian_spread,error_fun)
    test = obj_fun(params, x_train, y_train)
    result = minimize(
        obj_fun,
        params,
        args=(
            x_train,
            y_train,
        ),
        bounds=curve_fit_boundaries + distro_boundaries,
        method=method,
    )
    # Generate and plot the fitted kernel
    fitted_kernel = generate_kernel(distro, result.x[2:], kernel_width)
    train_err = obj_fun(result.x, x_train, y_train)
    y_pred = calc_its_convolution(x_test, fitted_kernel, *result.x[:2], los_cutoff)

    train_length = len(x_train)
    test_err = obj_fun(result.x, x_test[train_length-los_cutoff:], y_test[train_length-los_cutoff:])
    

    result_dict = {}
    result_dict["params"] = result.x
    result_dict["kernel"] = fitted_kernel
    result_dict["curve"] = y_pred
    result_dict["train_error"] = train_err
    result_dict["test_error"] = test_err
    result_dict["minimization_result"] = result
    return result_dict

# Objective function for direct kernel fit
def objective_direct_kernel_fit_with_previous_results(distro, kernel_length, los_cutoff, gaussian_spread=False,error_fun=mse):
    def objective_function(params, inc, icu, past_kernels):
        transition_rate, delay, *fun_params = params
        kernel = generate_kernel(distro, fun_params, kernel_length)
        kernels = np.zeros((len(past_kernels)+1,kernel_length))
        kernels[-1] = kernel
        observed = calc_its_convolution(inc, kernels, transition_rate, delay, los_cutoff,gaussian_spread)
        return error_fun(icu[los_cutoff:], observed[los_cutoff:])

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
    gaussian_spread=False,
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
    # obj_fun = objective_direct_kernel_fit_with_previous_results(distro, kernel_width, los_cutoff, gaussian_spread, error_fun)
    kernel_length = kernel_width
    def obj_fun(params, inc, icu, past_kernels, plot = False):
        transition_rate, delay, *fun_params = params
        kernels = stitch_together_multidim_kernel(past_kernels, fun_params)
        observed = calc_its_convolution(inc, kernels, transition_rate, delay, los_cutoff,gaussian_spread)
        # if plot:
        #     observed_without_cutoff = calc_its_convolution(inc, kernels, transition_rate, delay, 0,gaussian_spread)
        #     old_observed = calc_its_convolution(inc, past_kernels[0], transition_rate, delay, 0,gaussian_spread)
        #     plt.plot(icu,label="icu") 
        #     plt.plot(observed_without_cutoff,label="observed")
        #     plt.plot(old_observed,label="old_observed")
        #     plt.legend()
        #     plt.show()
        return error_fun(icu[los_cutoff:], observed[los_cutoff:])

    def stitch_together_multidim_kernel(past_kernels, fun_params):
        kernel = generate_kernel(distro, fun_params, kernel_length)
        kernels = np.zeros((len(past_kernels)+1,kernel_length))
        kernels[:len(past_kernels)] = past_kernels
        kernels[-1] = kernel
        return kernels
    test = obj_fun(params, x_train, y_train, past_kernels,plot=True)
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
    kernels = stitch_together_multidim_kernel(past_kernels, result.x[2:])

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
