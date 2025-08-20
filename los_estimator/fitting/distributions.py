"""Distribution types and probability density functions for LOS modeling.

This module provides various probability distributions that can be used as
kernels for length of stay estimation, including standard statistical
distributions and custom distributions specific to hospital data.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import (
    beta,
    cauchy,
    expon,
    gamma,
    invgauss,
    lognorm,
    norm,
    t,
    weibull_min,
)

__all__ = ["DistributionTypes", "Distribution"]


class DistributionTypes:
    """Enum for available distribution types.

    Defines constants for all supported distribution types that can be
    used as length of stay kernels in the modeling process.
    """

    LOGNORM = "lognorm"
    WEIBULL = "weibull"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"
    CAUCHY = "cauchy"
    T = "t"
    INVGAUSS = "invgauss"
    LINEAR = "linear"
    BLOCK = "block"
    SENTINEL = "sentinel"


@dataclass
class Distribution:
    """Data class for distribution information.

    Contains all necessary information to define and use a probability
    distribution for length of stay modeling.

    Attributes:
        name (str): Name identifier for the distribution.
        init_values (list): Initial parameter values for optimization.
        boundaries (list): Parameter bounds for constrained optimization.
        pdf (Callable): Probability density function.
    """

    name: str
    init_values: list
    boundaries: list
    pdf: Callable


# Sentinel data
sentinel_los_berlin = np.array(
    [
        0.01387985,
        0.04901323,
        0.0516157,
        0.05530254,
        0.04706137,
        0.05421817,
        0.05074821,
        0.04576014,
        0.03838647,
        0.03318152,
        0.03513338,
        0.02819345,
        0.03079592,
        0.02645847,
        0.02884407,
        0.02775971,
        0.01886792,
        0.01474734,
        0.01800043,
        0.01583171,
        0.01778356,
        0.01778356,
        0.00975927,
        0.01257862,
        0.01236174,
        0.01322923,
        0.01040989,
        0.0095424,
        0.00910865,
        0.0095424,
        0.00845804,
        0.00889178,
        0.0071568,
        0.00910865,
        0.01236174,
        0.00650618,
        0.00563869,
        0.00693993,
        0.00780742,
        0.00585556,
        0.00542182,
        0.00498807,
        0.00542182,
        0.00585556,
        0.00216873,
        0.00281935,
        0.00672305,
        0.00498807,
        0.00368684,
        0.00195185,
        0.00130124,
        0.00346996,
        0.00303622,
        0.00195185,
        0.00412058,
        0.0023856,
        0.00195185,
        0.0023856,
        0.00130124,
        0.00195185,
        0.00021687,
        0.00216873,
        0.00043375,
        0.00108436,
        0.00043375,
        0.00346996,
        0.00173498,
        0.00021687,
        0.00043375,
        0.00151811,
        0.00173498,
        0.00195185,
        0.00130124,
        0.00173498,
        0.00151811,
        0.00260247,
        0.00065062,
        0.00151811,
        0.0,
        0.00043375,
        0.00086749,
        0.00065062,
        0.00086749,
        0.00108436,
        0.00151811,
        0.00043375,
        0.00130124,
        0.00151811,
        0.0,
        0.00086749,
        0.00173498,
        0.00130124,
        0.0,
        0.0,
        0.00108436,
        0.0,
        0.00043375,
        0.00043375,
        0.0,
        0.00086749,
        0.00043375,
        0.00086749,
        0.0,
        0.0,
        0.0,
        0.00021687,
        0.0,
        0.0,
        0.0,
        0.00065062,
        0.0,
        0.00021687,
        0.00043375,
        0.00130124,
        0.0,
        0.00021687,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00065062,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00108436,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00086749,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00043375,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)


class DistributionsClass:
    """Collection of probability distributions for model fitting."""

    def generate_kernel(self, distro, fun_params, kernel_size):
        """Generate a kernel using the specified distribution."""
        *model_config, scaling_fac = fun_params
        pdf = self.get_pdf(distro)
        x = np.arange(kernel_size, dtype=float) * scaling_fac
        kernel = pdf(x, *model_config)
        result = kernel / kernel.sum()
        return result

    _distributions = {
        DistributionTypes.LOGNORM: Distribution(
            name=DistributionTypes.LOGNORM,
            init_values=[1, 0],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, sigma, μ: lognorm.pdf(x, s=sigma, scale=np.exp(μ)),
        ),
        DistributionTypes.WEIBULL: Distribution(
            name=DistributionTypes.WEIBULL,
            init_values=[1, 15],
            boundaries=[(1, None), (0, None)],
            pdf=lambda x, k, λ: weibull_min.pdf(x, c=k, scale=λ),
        ),
        DistributionTypes.GAUSSIAN: Distribution(
            name=DistributionTypes.GAUSSIAN,
            init_values=[0, 1],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, μ, sigma: norm.pdf(x, loc=μ, scale=sigma),
        ),
        DistributionTypes.EXPONENTIAL: Distribution(
            name=DistributionTypes.EXPONENTIAL,
            init_values=[1],
            boundaries=[(0.001, None)],
            pdf=lambda x, λ: expon.pdf(x, scale=1 / λ),
        ),
        DistributionTypes.GAMMA: Distribution(
            name=DistributionTypes.GAMMA,
            init_values=[2, 2],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, a, s: gamma.pdf(x, a=a, scale=s),
        ),
        DistributionTypes.BETA: Distribution(
            name=DistributionTypes.BETA,
            init_values=[2, 2],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, a, b: beta.pdf(x, a=a, b=b),
        ),
        DistributionTypes.CAUCHY: Distribution(
            name=DistributionTypes.CAUCHY,
            init_values=[0, 1],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, μ, s: cauchy.pdf(x, loc=μ, scale=s),
        ),
        DistributionTypes.T: Distribution(
            name=DistributionTypes.T,
            init_values=[10, 0, 1],
            boundaries=[(0, None), (0, None), (0, None)],
            pdf=lambda x, v, μ, s: t.pdf(x, df=v, loc=μ, scale=s),
        ),
        DistributionTypes.INVGAUSS: Distribution(
            name=DistributionTypes.INVGAUSS,
            init_values=[1, 0],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, μ, loc: invgauss.pdf(x, μ, loc=loc),
        ),
        DistributionTypes.LINEAR: Distribution(
            name=DistributionTypes.LINEAR,
            init_values=[40],
            boundaries=[(0, None)],
            pdf=lambda x, L: np.clip(-x / L + 1, 0, None),
        ),
        DistributionTypes.BLOCK: Distribution(
            name=DistributionTypes.BLOCK,
            init_values=[],
            boundaries=[],
            pdf=lambda x: np.eye(1, len(x), 1, dtype=float).ravel(),
        ),
        DistributionTypes.SENTINEL: Distribution(
            name=DistributionTypes.SENTINEL,
            init_values=[],
            boundaries=[],
            pdf=lambda x: np.asarray(sentinel_los_berlin, dtype=float),
        ),
    }

    def __getitem__(self, distro_name):
        """Get the distribution by name."""
        if distro_name in self._distributions:
            return self._distributions[distro_name]
        else:
            raise ValueError(f"Unknown Distribution: {distro_name}")

    def get_pdf(self, distro_name):
        """Returns the PDF function for the given distribution type."""
        return self[distro_name].pdf


Distributions = DistributionsClass()
