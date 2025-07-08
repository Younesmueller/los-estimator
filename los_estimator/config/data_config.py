"""Data configuration for LOS Estimator."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    los_file: str
    init_params_file: str
    mutants_file: Optional[str] = None
    start_day: str = "2020-01-01"
    end_day: str = "2025-01-01"
    sentinel_start_date: Optional[pd.Timestamp] = None
    sentinel_end_date: Optional[pd.Timestamp] = None
    
    def __post_init__(self):
        if self.sentinel_start_date is None:
            self.sentinel_start_date = pd.Timestamp("2020-10-01")
        if self.sentinel_end_date is None:
            self.sentinel_end_date = pd.Timestamp("2021-06-21")
