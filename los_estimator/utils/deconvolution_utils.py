"""Deconvolution-specific utility functions."""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple


def load_inc_beds(start_day, end_day):
    """Load combined incidence and bed occupancy data.
    
    Args:
        start_day: Start date for data loading
        end_day: End date for data loading
        
    Returns:
        pandas.DataFrame: Combined incidence and ICU data
    """
    from ..data.dataprep import load_incidences, load_hospitalizations
    
    df_inc, raw = load_incidences(start_day, end_day)
    
    # Remove duplicate 2020-01-01 entries if they exist
    if (len(df_inc) >= 2 and 
        df_inc.index[0] == pd.Timestamp("2020-01-01") and 
        df_inc.index[1] == pd.Timestamp("2020-01-01")):
        df_inc = df_inc.iloc[1:]

    df_icu = load_hospitalizations(start_day, end_day)
    df = df_inc.join(df_icu, how="inner")
    
    # Add smoothed new ICU admissions if column exists
    if "new_icu" in df.columns:
        df["new_icu_smooth"] = df["new_icu"].rolling(7).mean()
    
    return df


def generate_xticks(df: pd.DataFrame) -> Tuple[List[int], List[str]]:
    """Generate x-axis tick positions and labels for time series plots.
    
    Args:
        df (pd.DataFrame): DataFrame with datetime index
        
    Returns:
        tuple: (tick_positions, tick_labels)
    """
    xtick_pos = []
    xtick_label = []
    
    for i in range(len(df)):
        if df.index[i].day == 1:  # first day of month
            xtick_pos.append(i)
            label = df.index[i].strftime("%b")
            if i == 0 or df.index[i].month == 1:
                label += f"\n{df.index[i].year}"
            xtick_label.append(label)
    
    return xtick_pos, xtick_label


def create_time_windows(data_length: int, train_width: int, test_width: int, step: int):
    """Create sliding time windows for analysis.
    
    Args:
        data_length (int): Total length of the data
        train_width (int): Width of training window
        test_width (int): Width of test window  
        step (int): Step size between windows
        
    Returns:
        list: List of (train_start, train_end, test_start, test_end) tuples
    """
    windows = []
    current_pos = 0
    
    while current_pos + train_width + test_width <= data_length:
        train_start = current_pos
        train_end = current_pos + train_width
        test_start = train_end
        test_end = test_start + test_width
        
        windows.append((train_start, train_end, test_start, test_end))
        current_pos += step
    
    return windows


def calculate_relative_error(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Calculate relative error between predicted and actual values.
    
    Args:
        predicted (np.ndarray): Predicted values
        actual (np.ndarray): Actual values
        
    Returns:
        np.ndarray: Relative errors
    """
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs(predicted - actual) / np.abs(actual)
        rel_error[actual == 0] = 0  # Set error to 0 when actual is 0
        rel_error[np.isnan(rel_error)] = 0  # Handle NaN values
        rel_error[np.isinf(rel_error)] = 0  # Handle infinite values
    
    return rel_error


def smooth_data(data: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Apply smoothing to data using rolling mean.
    
    Args:
        data (np.ndarray): Input data to smooth
        window_size (int): Size of smoothing window
        
    Returns:
        np.ndarray: Smoothed data
    """
    if len(data) < window_size:
        return data
    
    # Convert to pandas Series for rolling mean
    series = pd.Series(data)
    smoothed = series.rolling(window=window_size, center=True, min_periods=1).mean()
    
    return smoothed.values


def validate_input_parameters(params_dict: dict) -> dict:
    """Validate input parameters for deconvolution analysis.
    
    Args:
        params_dict (dict): Dictionary of parameters to validate
        
    Returns:
        dict: Validation results with warnings and errors
    """
    results = {"warnings": [], "errors": []}
    
    required_params = ["kernel_width", "los_cutoff", "train_width", "test_width", "step"]
    
    # Check required parameters
    for param in required_params:
        if param not in params_dict:
            results["errors"].append(f"Missing required parameter: {param}")
        elif params_dict[param] <= 0:
            results["errors"].append(f"Parameter {param} must be positive")
    
    # Check parameter relationships
    if "train_width" in params_dict and "test_width" in params_dict:
        if params_dict["train_width"] < params_dict["test_width"]:
            results["warnings"].append("Train width is smaller than test width")
    
    if "kernel_width" in params_dict and "los_cutoff" in params_dict:
        if params_dict["kernel_width"] < params_dict["los_cutoff"]:
            results["warnings"].append("Kernel width is smaller than LOS cutoff")
    
    return results


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"
