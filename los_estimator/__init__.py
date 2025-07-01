"""
LOS Estimator Package

A Python package for estimating length of stay distributions in ICU settings
using deconvolution techniques on time series data.
"""

__version__ = "0.1.0"
__author__ = "LOS Estimator Team"

from .core.estimator import LOSEstimator
from .core.models import EstimationParams, EstimationResult
from .data.loader import DataLoader
from .visualization.plots import Visualizer

__all__ = [
    "LOSEstimator",
    "EstimationParams", 
    "EstimationResult",
    "DataLoader",
    "Visualizer"
]
