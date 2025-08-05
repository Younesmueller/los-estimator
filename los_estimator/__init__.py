"""Length of Stay Estimator Package

A package for analyzing hospital length of stay using deconvolution methods.
"""

# Core classes
from .core import *

# Configuration classes
from .config import DataConfig, ModelConfig, OutputConfig

# Data loading
from .data import DataLoader

# Fitting algorithms
from .fitting import MultiSeriesFitter
from .estimation_run import LosEstimationRun

# Visualization components
from .visualization import (
    VisualizationContext, 
    get_color_palette,
    InputDataVisualizer,
    DeconvolutionPlots,
    DeconvolutionAnimator,
    Visualizer
)


__version__ = "1.0.0"

__all__ = [
    "LosEstimationRun",

    # Core classes
    "ModelConfig",
    "WindowInfo", 
    "SeriesData",
    "SingleFitResult",
    "SeriesFitResult", 
    "MultiSeriesFitResults",
    "Utils",
    
    # Configuration
    "DataConfig",
    "ModelConfig", 
    "OutputConfig",
    
    # Data
    "DataLoader",
    
    # Fitting
    "MultiSeriesFitter",
    
    # Visualization
    "VisualizationContext",
    "get_color_palette",
    "InputDataVisualizer",
    "DeconvolutionPlots", 
    "DeconvolutionAnimator",
    "Visualizer",
    
    # Utils
    "compare_fit_results",
    "create_result_folders",
    "generate_run_name"
]