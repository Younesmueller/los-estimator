"""Data preparation utilities for LOS estimation."""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_los(cutoff_percentage=0.9, file=""):
    """Load length-of-stay distribution from CSV file.
    
    Args:
        cutoff_percentage (float): Cumulative percentage for cutoff determination
        file (str): Path to the LOS CSV file
        
    Returns:
        tuple: (los_distribution, los_cutoff)
    """
    df_los = pd.read_csv(file, index_col=0)
    
    los = df_los.iloc[:, 0].to_numpy(dtype=float)
    los /= los.sum()
    
    los_cutoff = np.argmax(los.cumsum() > cutoff_percentage)
    return los, los_cutoff


def load_incidences(start_day, end_day, data_dir="../data"):
    """Load incidence data from cases.csv file.
    
    Args:
        start_day: Start date for data filtering
        end_day: End date for data filtering
        data_dir (str): Directory containing the data files
        
    Returns:
        tuple: (filtered_dataframe, raw_dataframe)
    """
    file = os.path.join(data_dir, "cases.csv")
    df_inc = pd.read_csv(file, index_col=0, parse_dates=["Refdatum"])
    raw = df_inc.copy()
    df_inc = df_inc[["AnzahlFall", "daily"]]

    # Add missing dates at the beginning
    date_range = pd.date_range(start="2020-01-01", end=df_inc.index.min(), inclusive='left')
    new_data = pd.DataFrame(0, index=date_range, columns=df_inc.columns)
    df_inc = pd.concat([new_data, df_inc])
    
    # Filter by date range
    df_inc = df_inc[df_inc.index >= start_day]
    df_inc = df_inc[df_inc.index <= end_day]
    return df_inc, raw


def load_age_groups(start_day, end_day, data_dir="../data"):
    """Load age group data from Intensivregister file.
    
    Args:
        start_day: Start date for data filtering
        end_day: End date for data filtering
        data_dir (str): Directory containing the data files
        
    Returns:
        pandas.DataFrame: Age group data filtered by date range
    """
    file = os.path.join(data_dir, "Intensivregister_Deutschland_Altersgruppen.csv")
    df_age = pd.read_csv(file, parse_dates=["datum"])
    df_age.set_index("datum", inplace=True)
    df_age.drop(columns=["bundesland_id", "bundesland_name"], inplace=True)

    # Add missing dates at the beginning
    date_range = pd.date_range(start="2020-01-01", end=df_age.index.min())
    new_data = pd.DataFrame(0, index=date_range, columns=df_age.columns)
    df_age = pd.concat([new_data, df_age])
    
    # Filter by date range
    df_age = df_age[df_age.index >= start_day]
    df_age = df_age[df_age.index <= end_day]
    return df_age


def load_hospitalizations(start_day, end_day, data_dir="../data"):
    """Load hospitalization data.
    
    Args:
        start_day: Start date for data filtering
        end_day: End date for data filtering
        data_dir (str): Directory containing the data files
        
    Returns:
        pandas.DataFrame: Hospitalization data
    """
    file = os.path.join(data_dir, "hosp_ag.csv")
    if os.path.exists(file):
        df_hosp = pd.read_csv(file, index_col=0, parse_dates=True)
        df_hosp = df_hosp[df_hosp.index >= start_day]
        df_hosp = df_hosp[df_hosp.index <= end_day]
        return df_hosp
    else:
        print(f"Warning: Hospitalization file {file} not found")
        return pd.DataFrame()


def date_to_day(date, start_day):
    """Convert date to day number relative to start_day."""
    return (date - pd.Timestamp(start_day)).days


def day_to_date(day, start_day):
    """Convert day number to date relative to start_day."""
    return pd.Timestamp(start_day) + pd.Timedelta(days=day)


def prepare_time_series(data, start_day, end_day, fill_missing=True):
    """Prepare time series data with consistent date range.
    
    Args:
        data (pd.DataFrame): Input data with date index
        start_day: Start date
        end_day: End date
        fill_missing (bool): Whether to fill missing dates with zeros
        
    Returns:
        pd.DataFrame: Prepared time series data
    """
    if fill_missing:
        # Create full date range
        date_range = pd.date_range(start=start_day, end=end_day)
        data = data.reindex(date_range, fill_value=0)
    else:
        # Just filter by date range
        data = data[data.index >= start_day]
        data = data[data.index <= end_day]
    
    return data


def validate_data_consistency(incidences, hospitalizations, age_groups):
    """Validate that all datasets have consistent date ranges and no missing values.
    
    Args:
        incidences (pd.DataFrame): Incidence data
        hospitalizations (pd.DataFrame): Hospitalization data
        age_groups (pd.DataFrame): Age group data
        
    Returns:
        dict: Validation results with warnings and errors
    """
    results = {"warnings": [], "errors": []}
    
    # Check date ranges
    datasets = {"incidences": incidences, "hospitalizations": hospitalizations, "age_groups": age_groups}
    date_ranges = {}
    
    for name, df in datasets.items():
        if not df.empty:
            date_ranges[name] = (df.index.min(), df.index.max())
    
    # Check for consistent date ranges
    if len(set(date_ranges.values())) > 1:
        results["warnings"].append("Datasets have different date ranges")
        for name, (start, end) in date_ranges.items():
            results["warnings"].append(f"{name}: {start} to {end}")
    
    # Check for missing values
    for name, df in datasets.items():
        if not df.empty:
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                results["warnings"].append(f"{name} has {missing_count} missing values")
    
    return results
