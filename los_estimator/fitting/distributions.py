import numpy as np
from scipy.stats import lognorm, weibull_min, norm, expon, gamma, beta, cauchy, t, invgauss


def generate_kernel(distro, fun_params, kernel_size):
    *params, scaling_fac = fun_params
    pdf = DISTROS[distro]
    x = np.arange(kernel_size, dtype=float) * scaling_fac
    kernel = pdf(x, *params)
    result = kernel / kernel.sum()
    return result

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
