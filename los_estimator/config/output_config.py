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
