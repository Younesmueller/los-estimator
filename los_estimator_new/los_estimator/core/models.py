"""
Data structures and models for LOS estimation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import functools



class ErrorFunction(Enum):
    """Available error functions for optimization."""
    MSE = "mse"
    WEIGHTED_MSE = "weighted_mse"
    MAE = "mae"
    REL_ERR = "rel_err"
    INV_REL_ERR = "inv_rel_err"
    CAPACITY_ERR = "capacity_err"


@dataclass
class EstimationParams:
    """Parameters for LOS estimation."""
    
    # Core parameters
    kernel_width: int = 120
    los_cutoff: int = 60
    train_width: int = 102  # 42 + los_cutoff
    test_width: int = 21
    step: int = 7
    
    # Data processing
    fit_admissions: bool = True
    smooth_data: bool = False
    
    # Optimization
    error_function: ErrorFunction = ErrorFunction.MSE
    reuse_last_parametrization: bool = True
    variable_kernels: bool = True
    
    # Time range
    start_day: str = "2020-01-01"
    end_day: str = "2025-01-01"
    
    def __post_init__(self):
        """Validate and adjust parameters after initialization."""
        if isinstance(self.error_function, str):
            self.error_function = ErrorFunction(self.error_function)
        
        # Adjust train_width to include los_cutoff if needed
        if self.train_width < self.los_cutoff + 42:
            self.train_width = self.los_cutoff + 42


class WindowInfo:
    """Information about a sliding window for estimation."""
    
    def __init__(self, window: int, params: EstimationParams):
        self.window = window
        self.train_end = window
        self.train_start = window - params.train_width
        self.train_los_cutoff = self.train_start + params.los_cutoff
        self.test_start = self.train_end
        self.test_end = self.test_start + params.test_width
        
        # Create slices for easy indexing
        self.train_window = slice(self.train_start, self.train_end)
        self.train_test_window = slice(self.train_start, self.test_end)
        self.test_window = slice(self.test_start, self.test_end)
    
    def __repr__(self):
        return (f"WindowInfo(window={self.window}, "
                f"train_start={self.train_start}, train_end={self.train_end}, "
                f"test_start={self.test_start}, test_end={self.test_end})")


class SeriesData:
    """Container for time series data with windowing functionality."""
    
    def __init__(self, df_occupancy: pd.DataFrame, params: EstimationParams):
        self.params = params
        self.x_full, self.y_full = self._select_series(df_occupancy, params)
        self.n_days = len(self.x_full)
        self._calc_windows(params)
    
    def _select_series(self, df: pd.DataFrame, params: EstimationParams) -> Tuple[np.ndarray, np.ndarray]:
        """Select appropriate series based on parameters."""
        if params.fit_admissions:
            col = "new_icu_smooth" if params.smooth_data else "new_icu"
        else:
            col = "AnzahlFall" if params.smooth_data else "daily"
        
        return df[col].values, df["icu"].values
    
    def _calc_windows(self, params: EstimationParams):
        """Calculate sliding windows for estimation."""
        start = 0
        if params.fit_admissions:
            # This would need to be injected or calculated differently
            # For now, using a reasonable default
            start = 50 + params.train_width
        
        self.windows = np.arange(start, len(self.x_full) - params.kernel_width, params.step)
        self.window_infos = [WindowInfo(window, params) for window in self.windows]
        self.n_windows = len(self.windows)
    
    @functools.lru_cache(maxsize=None)
    def get_train_data(self, window_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data for a specific window."""
        if window_id >= len(self.windows):
            raise ValueError(f"Window ID {window_id} out of range for {len(self.windows)} windows.")
        
        w = self.window_infos[window_id]
        return self.x_full[w.train_window], self.y_full[w.train_window]
    
    @functools.lru_cache(maxsize=None)
    def get_test_data(self, window_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data for a specific window."""
        if window_id >= len(self.windows):
            raise ValueError(f"Window ID {window_id} out of range for {len(self.windows)} windows.")
        
        w = self.window_infos[window_id]
        return self.x_full[w.train_test_window], self.y_full[w.train_test_window]
    
    @functools.lru_cache(maxsize=None)
    def get_window_info(self, window_id: int) -> WindowInfo:
        """Get window information for a specific window."""
        if window_id >= len(self.windows):
            raise ValueError(f"Window ID {window_id} out of range for {len(self.windows)} windows.")
        
        return self.window_infos[window_id]
    
    def __iter__(self):
        """Iterate over all windows."""
        for idx in range(self.n_windows):
            train_data = self.get_train_data(idx)
            test_data = self.get_test_data(idx)
            window_info = self.get_window_info(idx)
            yield idx, window_info, train_data, test_data
    
    def __len__(self):
        return self.n_windows
    
    def __repr__(self):
        return (f"SeriesData(n_windows={self.n_windows}, "
                f"kernel_width={self.params.kernel_width}, "
                f"los_cutoff={self.params.los_cutoff})")


class SingleFitResult:
    """Result of a single window fitting operation."""
    
    def __init__(self, success: bool = False, params: Optional[List[float]] = None, 
                 rel_train_error: float = np.inf, rel_test_error: float = np.inf):
        self.success = success
        self.params = params or []
        self.rel_train_error = rel_train_error
        self.rel_test_error = rel_test_error


class SeriesFitResult:
    """Results for fitting a single distribution across all windows."""
    
    def __init__(self, distro: str):
        self.distro = distro
        self.window_infos: List[WindowInfo] = []
        self.fit_results: List[Optional[SingleFitResult]] = []
        self.train_relative_errors: Optional[np.ndarray] = None
        self.test_relative_errors: Optional[np.ndarray] = None
        self.successes: List[bool] = []
        self.n_success: int = 0
        self.transition_rates: Optional[np.ndarray] = None
        self.transition_delays: Optional[np.ndarray] = None
    
    def append(self, window_info: WindowInfo, fit_result: SingleFitResult):
        """Add a new fit result."""
        if not isinstance(window_info, WindowInfo):
            raise TypeError("window_info must be an instance of WindowInfo")
        if not isinstance(fit_result, SingleFitResult):
            raise TypeError("fit_result must be an instance of SingleFitResult")
        
        self.window_infos.append(window_info)
        self.fit_results.append(fit_result)
    
    def bake(self):
        """Finalize the results and compute aggregate statistics."""
        self._collect_errors()
        self.successes = [fr.success if fr else False for fr in self.fit_results]
        self.n_success = sum(self.successes)
        
        self.transition_rates = np.array([
            fr.params[0] if (fr and fr.success and len(fr.params) > 0) else np.nan 
            for fr in self.fit_results
        ])
        
        self.transition_delays = np.array([
            fr.params[1] if (fr and fr.success and len(fr.params) > 1) else np.nan 
            for fr in self.fit_results
        ])
        
        return self
    
    def _collect_errors(self):
        """Collect training and test errors."""
        train_err = np.empty(len(self.fit_results))
        test_err = np.empty(len(self.fit_results))
        
        for i, fr in enumerate(self.fit_results):
            if fr is None:
                train_err[i] = np.inf
                test_err[i] = np.inf
            else:
                train_err[i] = fr.rel_train_error
                test_err[i] = fr.rel_test_error
        
        self.train_relative_errors = train_err
        self.test_relative_errors = test_err
    
    def __getitem__(self, window_id):
        """Get fit result for a specific window."""
        if isinstance(window_id, slice):
            return self.fit_results[window_id]
        if window_id >= len(self.fit_results):
            raise IndexError(f"Window ID {window_id} out of range for {len(self.fit_results)} windows.")
        return self.fit_results[window_id]
    
    def __setitem__(self, window_id, value):
        """Set fit result for a specific window."""
        if window_id >= len(self.fit_results):
            raise IndexError(f"Window ID {window_id} out of range for {len(self.fit_results)} windows.")
        self.fit_results[window_id] = value
    
    def __repr__(self):
        return (f"SeriesFitResult(distro={self.distro}, n_windows={len(self.window_infos)}, "
                f"n_success={self.n_success})")


class EstimationResult:
    """Complete estimation result containing all fitted distributions."""
    
    def __init__(self, series_results: Dict[str, SeriesFitResult], 
                 params: EstimationParams, metadata: Optional[Dict[str, Any]] = None):
        self.series_results = series_results
        self.params = params
        self.metadata = metadata or {}
        self.distros = list(series_results.keys())
    
    def get_best_distribution(self, metric: str = "test_error") -> str:
        """Get the best performing distribution based on specified metric."""
        if metric == "test_error":
            best_distro = min(self.distros, 
                            key=lambda d: np.nanmean(self.series_results[d].test_relative_errors))
        elif metric == "train_error":
            best_distro = min(self.distros, 
                            key=lambda d: np.nanmean(self.series_results[d].train_relative_errors))
        elif metric == "success_rate":
            best_distro = max(self.distros, 
                            key=lambda d: self.series_results[d].n_success)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_distro
    
    @functools.lru_cache(maxsize=None)
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics for all distributions."""
        stats = []
        
        for distro in self.distros:
            result = self.series_results[distro]
            stats.append({
                'distribution': distro,
                'success_rate': result.n_success / len(result.fit_results),
                'mean_train_error': np.nanmean(result.train_relative_errors),
                'mean_test_error': np.nanmean(result.test_relative_errors),
                'std_train_error': np.nanstd(result.train_relative_errors),
                'std_test_error': np.nanstd(result.test_relative_errors)
            })
        
        return pd.DataFrame(stats)
    
    def __repr__(self):
        return f"EstimationResult(distros={self.distros}, n_windows={len(self.series_results[self.distros[0]].fit_results) if self.distros else 0})"
