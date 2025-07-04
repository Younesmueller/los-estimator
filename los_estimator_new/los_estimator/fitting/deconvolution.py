"""
Deconvolution algorithms for LOS estimation.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize
from numba import njit

from ..core.models import (
    EstimationParams, SeriesData, WindowInfo, SingleFitResult, 
    SeriesFitResult, EstimationResult, ErrorFunction
)
from .distributions import DistributionFitter


class DeconvolutionEngine:
    """Core deconvolution engine for LOS estimation."""
    
    def __init__(self, distribution_fitter: Optional[DistributionFitter] = None):
        """
        Initialize deconvolution engine.
        
        Args:
            distribution_fitter: Distribution fitting utility
        """
        self.distribution_fitter = distribution_fitter or DistributionFitter()
    
    def fit_single_window(self, train_data: Tuple[np.ndarray, np.ndarray],
                         test_data: Tuple[np.ndarray, np.ndarray],
                         distro_name: str, initial_params: Optional[List[float]] = None,
                         params: Optional[EstimationParams] = None) -> SingleFitResult:
        """
        Fit a single window using deconvolution.
        
        Args:
            train_data: Training data (x, y)
            test_data: Test data (x, y) 
            distro_name: Distribution name to fit
            initial_params: Initial parameters for optimization
            params: Estimation parameters
            
        Returns:
            SingleFitResult
        """
        if params is None:
            params = EstimationParams()
        
        x_train, y_train = train_data
        x_test, y_test = test_data
        
        try:
            # Set up optimization problem
            if initial_params is None:
                initial_params = self._get_initial_params(distro_name)
            
            # Define objective function
            def objective(opt_params):
                try:
                    # Extract parameters
                    transition_rate = opt_params[0]
                    transition_delay = opt_params[1] if len(opt_params) > 1 else 0
                    distro_params = opt_params[2:] if len(opt_params) > 2 else []
                    
                    # Generate LOS distribution
                    los_dist = self.distribution_fitter.generate_distribution_samples(
                        distro_name, distro_params, params.los_cutoff
                    )
                    
                    # Perform convolution
                    predicted = self._convolve_series(x_train, los_dist, transition_rate, transition_delay)
                    
                    # Calculate error based on selected error function
                    if params.error_function == ErrorFunction.MSE:
                        error = np.mean((predicted - y_train) ** 2)
                    elif params.error_function == ErrorFunction.WEIGHTED_MSE:
                        weights = np.exp(np.linspace(0, 2, len(predicted)))
                        weights /= weights.sum()
                        error = np.sum(((predicted - y_train) ** 2) * weights)
                    else:
                        error = np.mean(np.abs(predicted - y_train))
                    
                    return error
                    
                except Exception as e:
                    print (f"Error in objective function: {e}")
                    return np.inf
            
            if not params.fit_admissions:
                raise ValueError("Fitting admissions is disabled. Cannot fit distribution.")
            else:
                # Optimization bounds
                bounds = self._get_parameter_bounds(distro_name, len(initial_params))
            
            # Perform optimization
            result = minimize(objective, initial_params, method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 500})
            XXX> Hiernicht abbrechen
            if result.success and result.fun < np.inf:
                # Calculate test error
                opt_params = result.x
                transition_rate = opt_params[0]
                transition_delay = opt_params[1] if len(opt_params) > 1 else 0
                distro_params = opt_params[2:] if len(opt_params) > 2 else []
                
                # Generate final LOS distribution
                los_dist = self.distribution_fitter.generate_distribution_samples(
                    distro_name, distro_params, params.los_cutoff
                )
                
                # Test prediction
                predicted_test = self._convolve_series(x_test, los_dist, transition_rate, transition_delay)
                test_error = np.mean((predicted_test - y_test) ** 2)
                
                # Calculate relative errors
                train_rel_error = result.fun / (np.mean(y_train) ** 2 + 1e-8)
                test_rel_error = test_error / (np.mean(y_test) ** 2 + 1e-8)
                
                return SingleFitResult(
                    success=True,
                    params=opt_params.tolist(),
                    rel_train_error=train_rel_error,
                    rel_test_error=test_rel_error
                )
            else:
                return SingleFitResult(success=False)
                
        except Exception:
            return SingleFitResult(success=False)
    
    def fit_distribution_across_windows(self, series_data: SeriesData, distro_name: str,
                                      initial_params_df: Optional[Dict] = None,
                                      params: Optional[EstimationParams] = None) -> SeriesFitResult:
        """
        Fit a single distribution across all windows.
        
        Args:
            series_data: Time series data with windows
            distro_name: Distribution name to fit
            initial_params_df: Dictionary of initial parameters
            params: Estimation parameters
            
        Returns:
            SeriesFitResult containing all window results
        """
        if params is None:
            params = EstimationParams()
        
        result = SeriesFitResult(distro_name)
        
        for window_id, window_info, train_data, test_data in series_data:
            # Get initial parameters
            initial_params = self._get_initial_params_for_window(
                distro_name, window_id, result, initial_params_df, params
            )
            

            # Fit single window
            fit_result = self.fit_single_window(
                train_data, test_data, distro_name, initial_params, params
            )
            
            result.append(window_info, fit_result)
        
        return result.bake()
    
    def fit_multiple_distributions(self, series_data: SeriesData, 
                                 distribution_names: List[str],
                                 initial_params_df: Optional[Dict] = None,
                                 params: Optional[EstimationParams] = None) -> EstimationResult:
        """
        Fit multiple distributions across all windows.
        
        Args:
            series_data: Time series data with windows
            distribution_names: List of distribution names
            initial_params_df: Dictionary of initial parameters
            params: Estimation parameters
            
        Returns:
            EstimationResult containing all results
        """
        if params is None:
            params = EstimationParams()
        
        series_results = {}
        
        for distro_name in distribution_names:
            print(f"Fitting {distro_name}...")
            series_results[distro_name] = self.fit_distribution_across_windows(
                series_data, distro_name, initial_params_df, params
            )
        
        return EstimationResult(series_results, params)
    
    def _convolve_series(self, admissions: np.ndarray, los_dist: np.ndarray, 
                        transition_rate: float, transition_delay: float) -> np.ndarray:
        """
        Convolve admissions with LOS distribution.
        
        Args:
            admissions: Admission time series
            los_dist: Length of stay distribution
            transition_rate: Transition rate parameter
            transition_delay: Transition delay parameter
            
        Returns:
            Convolved time series
        """

        los_dist = 1- np.cumprod(los_dist)
        # Simple convolution implementation
        # In practice, this would use the more sophisticated models from the original code
        result = np.convolve(admissions, los_dist, mode='same')
        
        # Apply transition parameters (simplified)
        result = result * (1 + transition_rate) 
        if transition_delay > 0:
            # Apply delay (simplified shift)
            delay_samples = int(transition_delay)
            if delay_samples > 0:
                result = np.roll(result, delay_samples)
                result[:delay_samples] = 0
        
        return result
    
    def _get_initial_params(self, distro_name: str) -> List[float]:
        """Get default initial parameters for a distribution."""
        # Add distribution-specific parameters
        distro_params = self.distribution_fitter._get_default_params(distro_name)
        
        return self.distribution_fitter.INIT_VALUES + distro_params
    
    def _get_initial_params_for_window(self, distro_name: str, window_id: int,
                                     series_result: SeriesFitResult,
                                     initial_params_df: Optional[Dict],
                                     params: EstimationParams) -> List[float]:
        """Get initial parameters for a specific window."""

        if not params.fit_admissions:
            raise ValueError("Fitting admissions is disabled. Cannot get initial parameters.")
        
        # Try to reuse last parametrization if enabled
        if params.reuse_last_parametrization and window_id > 0:
            for prev_id in range(window_id - 1, -1, -1):
                if prev_id < len(series_result.fit_results):
                    prev_result = series_result.fit_results[prev_id]
                    if prev_result and prev_result.success:
                        return prev_result.params
        
        # Fallback to initial parameters from dataframe
        if initial_params_df and distro_name in initial_params_df:
            return self.distribution_fitter.INIT_VALUES + initial_params_df[distro_name]
        
        # Ultimate fallback to defaults
        return self._get_initial_params(distro_name)
    
    def _get_parameter_bounds(self, distro_name: str, n_params: int) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        
        # Transition rate bounds
        bounds.append((1.,1.))
        
        # Transition delay bounds
        bounds.append((0.,0.,))
        
        # Distribution parameter bounds (simplified)
        for i in range(2, n_params):
            bounds.append((0.01, 100))  # Reasonable positive bounds
        
        return bounds
