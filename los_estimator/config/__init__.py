"""Model configuration for LOS Estimator."""

from dataclasses import dataclass
from typing import List
import types


from dataclasses import dataclass
from typing import Optional
import pandas as pd
from pathlib import Path



@dataclass
class ModelConfig:
    kernel_width: int
    los_cutoff: int
    smooth_data: bool
    train_width: int
    test_width: int
    step: int    
    error_fun: str
    reuse_last_parametrization: bool
    variable_kernels: bool
    distributions: list[str]
    ideas = types.SimpleNamespace(
        los_change_penalty=["..."],
        fitting_err=["mse", "mae", "rel_err", "weighted_mse", "inv_rel_err", "capacity_err", "..."],
        presenting_err=["..."]
    )

default_params = ModelConfig(
    kernel_width=120,
    los_cutoff=60,
    smooth_data=False,
    train_width=42 + 60,
    test_width=21, 
    step=7,
    error_fun="mse",
    reuse_last_parametrization=True,
    variable_kernels=True,
    distributions=[
         "lognorm",
        # "weibull",
        "gaussian",
        "exponential",
        # "gamma",
        # "beta",
        "cauchy",
        "t",
        # "invgauss",
        "linear",
        # "block",
        # "sentinel",
        "compartmental",
    ]
)


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
        

        
"""Output configuration for LOS Estimator."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutputConfig:
    """Configuration for output and visualization."""
    results_folder_base: str = "results"
    save_figs: bool = True
    show_figs: bool = True
    save_animations: bool = False
    dpi: int = 150
    figsize: tuple = (12, 8)
    
    def get_results_folder(self, run_name: str) -> Path:
        """Get the full results folder path for a run."""
        return Path(self.results_folder_base) / run_name
    
    def get_figures_folder(self, run_name: str) -> Path:
        """Get the figures folder path for a run."""
        return self.get_results_folder(run_name) / "figures"
    
    def get_animations_folder(self, run_name: str) -> Path:
        """Get the animations folder path for a run."""
        return self.get_results_folder(run_name) / "animations"
class DebugConfiguration:
    def __init__(self, one_window=False, less_windows=False, less_distros=False, only_linear=False):
        self.ONE_WINDOW = one_window
        self.LESS_WINDOWS = less_windows
        self.LESS_DISTROS = less_distros
        self.ONLY_LINEAR = only_linear

        

@dataclass
class OutputFolderConfig:
    base: str
    run_name: str = "default"

    def build(self):
        self.base = Path(self.base)
        self.results = self.base / self.run_name
        self.figures = self.results / "figures"
        self.animation = self.results / "animation"
    def __post_init__(self):
        self.build()



    