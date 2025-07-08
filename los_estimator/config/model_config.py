"""Model configuration for LOS Estimator."""

from dataclasses import dataclass
from typing import List


@dataclass 
class ModelConfig:
    """Configuration for model parameters."""
    kernel_width: int = 120
    los_cutoff: int = 60
    smooth_data: bool = False
    train_width: int = None
    test_width: int = 21
    step: int = 7
    fit_admissions: bool = True
    error_fun: str = "mse"
    reuse_last_parametrization: bool = True
    variable_kernels: bool = True
    distributions: List[str] = None
    
    def __post_init__(self):
        if self.train_width is None:
            self.train_width = 42 + self.los_cutoff
        if self.distributions is None:
            self.distributions = ["gaussian", "exponential", "linear", "compartmental"]
