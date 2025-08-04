"""Data configuration for LOS Estimator."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    cases_file: str | Path
    icu_occupancy_file: str | Path
    los_file: str | Path
    init_params_file: str | Path
    start_day: str | Path
    end_day: str | Path
    
    base_path: str | Path ="."
    mutants_file: Optional[str] = None
    sentinel_start_date: Optional[pd.Timestamp] = None
    sentinel_end_date: Optional[pd.Timestamp] = None
    
    def __post_init__(self):
        if self.sentinel_start_date is None:
            self.sentinel_start_date = pd.Timestamp("2020-10-01")
        if self.sentinel_end_date is None:
            self.sentinel_end_date = pd.Timestamp("2021-06-21")

        base_path = Path(self.base_path)
        self.cases_file = base_path / self.cases_file
        self.icu_occupancy_file = base_path / self.icu_occupancy_file
        self.los_file = base_path / self.los_file
        self.init_params_file = base_path / self.init_params_file

        if self.mutants_file is not None:
            self.mutants_file = base_path / self.mutants_file
        

        
