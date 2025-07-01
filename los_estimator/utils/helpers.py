"""
Utility functions and helpers for LOS estimation.
"""

import time
import shutil
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional
from ..core.models import EstimationParams


def generate_run_name(params: EstimationParams) -> str:
    """
    Generate a unique run name based on parameters.
    
    Args:
        params: Estimation parameters
        
    Returns:
        Unique run name string
    """
    timestamp = time.strftime("%y%m%d_%H%M")
    run_name = f"{timestamp}_deconv"
    
    # Add key parameters to name
    run_name += f"_k{params.kernel_width}"
    run_name += f"_tr{params.train_width}"
    run_name += f"_te{params.test_width}"
    run_name += f"_s{params.step}"
    
    # Add data type
    if params.fit_admissions:
        run_name += "_adm"
    else:
        run_name += "_inc"
    
    # Add smoothing info
    if params.smooth_data:
        run_name += "_smooth"
    else:
        run_name += "_raw"
    
    return run_name


def setup_directories(run_name: str, base_dir: str = "results") -> Tuple[Path, Path, Path]:
    """
    Set up directory structure for results.
    
    Args:
        run_name: Name of the run
        base_dir: Base directory for results
        
    Returns:
        Tuple of (results_folder, figures_folder, animation_folder)
    """
    base_path = Path(base_dir)
    results_folder = base_path / run_name
    figures_folder = results_folder / "figures"
    animation_folder = results_folder / "animations"
    
    # Create directories
    results_folder.mkdir(parents=True, exist_ok=True)
    figures_folder.mkdir(parents=True, exist_ok=True)
    animation_folder.mkdir(parents=True, exist_ok=True)
    
    return results_folder, figures_folder, animation_folder


def date_to_day(date: pd.Timestamp, start_day: str = "2020-01-01") -> int:
    """
    Convert date to day number since start_day.
    
    Args:
        date: Date to convert
        start_day: Reference start date
        
    Returns:
        Number of days since start_day
    """
    return (date - pd.Timestamp(start_day)).days


def day_to_date(day: int, start_day: str = "2020-01-01") -> pd.Timestamp:
    """
    Convert day number to date.
    
    Args:
        day: Day number since start_day
        start_day: Reference start date
        
    Returns:
        Date corresponding to day number
    """
    return pd.Timestamp(start_day) + pd.Timedelta(days=day)


def backup_original_files(source_dir: str, backup_dir: str) -> None:
    """
    Create backup of original files.
    
    Args:
        source_dir: Source directory to backup
        backup_dir: Backup destination directory
    """
    source_path = Path(source_dir)
    backup_path = Path(backup_dir)
    
    if source_path.exists():
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(source_path, backup_path)


def validate_data_files(file_paths: dict) -> dict:
    """
    Validate that required data files exist.
    
    Args:
        file_paths: Dictionary of file descriptions to paths
        
    Returns:
        Dictionary of validation results
    """
    results = {}
    
    for description, path in file_paths.items():
        file_path = Path(path)
        results[description] = {
            'path': str(path),
            'exists': file_path.exists(),
            'is_file': file_path.is_file() if file_path.exists() else False,
            'size_mb': file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
        }
    
    return results


def get_default_file_paths(base_dir: Optional[str] = None) -> dict:
    """
    Get default file paths for LOS estimation.
    
    Args:
        base_dir: Base directory (if None, uses relative paths)
        
    Returns:
        Dictionary of default file paths
    """
    if base_dir:
        base_path = Path(base_dir)
    else:
        base_path = Path(".")
    
    return {
        'los_file': base_path / "los-estimator" / "01_create_los_profiles" / "berlin" / "output_los" / "los_berlin_all.csv",
        'init_params_file': base_path / "los-estimator" / "02_fit_los_distributions" / "output_los" / "los_berlin_all" / "fit_results.csv",
        'incidence_file': base_path / "los-estimator" / "data" / "cases.csv",
        'icu_file': base_path / "los-estimator" / "data" / "Intensivregister_Bundeslaender_Kapazitaeten.csv",
        'mutants_file': base_path / "los-estimator" / "data" / "VOC_VOI_Tabelle.xlsx",
    }
