"""File and folder utilities."""

import time
from pathlib import Path
from typing import Tuple


def generate_run_name(params) -> str:
    """Generate a unique run name based on parameters.
    
    Args:
        params: Parameter object with model configuration
        
    Returns:
        Generated run name string
    """
    # Format: DDMMYY_HHMM_deconv_k{kernel_width}_tr{train_width}_te{test_width}_s{step}_{data_type}
    timestamp = time.strftime("%d%m%y_%H%M")
    
    # Extract key parameters
    kernel_width = getattr(params, 'kernel_width', 120)
    train_width = getattr(params, 'train_width', 102)
    test_width = getattr(params, 'test_width', 21)
    step = getattr(params, 'step', 7)
    
    # Determine data type
    if getattr(params, 'fit_admissions', False):
        data_type = "adm"
    else:
        data_type = "inc"
    
    # Determine smoothing
    if getattr(params, 'smooth_data', False):
        data_type += "_smooth"
    else:
        data_type += "_raw"
    
    run_name = f"{timestamp}_deconv_k{kernel_width}_tr{train_width}_te{test_width}_s{step}_{data_type}"
    
    return run_name


def create_result_folders(run_name: str, base_path: str = "results") -> Tuple[Path, Path, Path]:
    """Create result folders for a run.
    
    Args:
        run_name: Name of the run
        base_path: Base path for results
        
    Returns:
        Tuple of (results_folder, figures_folder, animation_folder)
    """
    results_folder = Path(base_path) / run_name
    figures_folder = results_folder / "figures"
    animation_folder = results_folder / "animations"
    
    # Create directories
    results_folder.mkdir(parents=True, exist_ok=True)
    figures_folder.mkdir(parents=True, exist_ok=True)
    animation_folder.mkdir(parents=True, exist_ok=True)
    
    return results_folder, figures_folder, animation_folder
