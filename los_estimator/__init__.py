"""Length of Stay Estimator Package

A package for analyzing hospital length of stay using deconvolution methods.
"""

import logging
from pathlib import Path


def setup_logging(log_file_path=None):
    """Configure logging for the whole package."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("los_estimator")

    if log_file_path:
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)
    return logger


setup_logging("current.log")

# Core classes
from .core import *

# Configuration classes
from .config import DataConfig, ModelConfig, VisualizationContext

# Data loading
from .data import DataLoader

# Fitting algorithms
from .fitting import MultiSeriesFitter
from .estimation_run import LosEstimationRun

# Visualization components
from .visualization import (
    InputDataVisualizer,
    DeconvolutionPlots,
    DeconvolutionAnimator,
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
    "ModelConfig",
    "DataConfig",
    "DebugConfig",
    "OutputFolderConfig",
    "AnimationConfig",
    "VisualizationConfig",
    "VisualizationContext",
    # Data
    "DataLoader",
    # Fitting
    "MultiSeriesFitter",
    # Visualization
    "InputDataVisualizer",
    "DeconvolutionPlots",
    "DeconvolutionAnimator",
]
