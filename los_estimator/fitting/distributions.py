"""
Distribution fitting utilities for LOS estimation.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, weibull_min, norm, expon, gamma, beta
from typing import Dict, List, Tuple, Optional, Callable
from numba import njit
from ..core.models import SingleFitResult, ErrorFunction


@njit
def mse(x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error."""
    return np.mean((x - y) ** 2)


@njit
def weighted_mse(x: np.ndarray, y: np.ndarray) -> float:
    """Weighted mean squared error with exponential weights."""
    le = len(x)
    weights = np.exp(np.linspace(0, 2, le))
    weights /= weights.sum()
    return np.sum(((x - y) ** 2) * weights)


@njit
def mae(x: np.ndarray, y: np.ndarray) -> float:
    """Mean absolute error."""
    return np.mean(np.abs(x - y))


# Sentinel LOS distribution for Berlin (from original code)
SENTINEL_LOS_BERLIN = np.array([
    0.01387985, 0.04901323, 0.0516157, 0.05530254, 0.04706137,
    0.05421817, 0.05074821, 0.04576014, 0.03838647, 0.03318152,
    0.03513338, 0.02819345, 0.03079592, 0.02645847, 0.02884407,
    0.02775971, 0.01886792, 0.01474734, 0.01800043, 0.01583171,
    0.01778356, 0.01778356, 0.00975927, 0.01257862, 0.01236174,
    0.01322923, 0.01040989, 0.0095424, 0.00910865, 0.0095424,
    0.00845804, 0.00889178, 0.0071568, 0.00910865, 0.01236174,
    0.00650618, 0.00563869, 0.00693993, 0.00780742, 0.00585556,
    0.00542182, 0.00498807, 0.00542182, 0.00585556, 0.00216873,
    0.00281935, 0.00672305, 0.00498807, 0.00368684, 0.00195185,
    0.00130124, 0.00346996, 0.00303622, 0.00195185, 0.00412058,
    0.0023856, 0.00195185, 0.0023856, 0.00130124, 0.00195185
])


class DistributionFitter:
    """Fits probability distributions to length of stay data."""
    
    # Available distributions
    DISTRIBUTIONS = {
        'lognorm': lognorm,
        'weibull': weibull_min,
        'normal': norm,
        'exponential': expon,
        'gamma': gamma,
        'beta': beta
    }
    
    ERROR_FUNCTIONS = {
        ErrorFunction.MSE: mse,
        ErrorFunction.WEIGHTED_MSE: weighted_mse,
        ErrorFunction.MAE: mae,
    }
    
    def __init__(self, max_los_days: int = 150):
        """
        Initialize distribution fitter.
        
        Args:
            max_los_days: Maximum length of stay in days
        """
        self.max_los_days = max_los_days
        self._days = np.arange(1, max_los_days + 1)
    
    def fit_distribution(self, distro_name: str, initial_params: Optional[List[float]] = None,
                        target_data: Optional[np.ndarray] = None) -> SingleFitResult:
        """
        Fit a specific distribution.
        
        Args:
            distro_name: Name of distribution to fit
            initial_params: Initial parameters for optimization
            target_data: Target data to fit to (if None, uses sentinel data)
            
        Returns:
            SingleFitResult with fit results
        """
        if distro_name not in self.DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution: {distro_name}")
        
        if target_data is None:
            target_data = SENTINEL_LOS_BERLIN[:min(len(SENTINEL_LOS_BERLIN), self.max_los_days)]
        
        distribution = self.DISTRIBUTIONS[distro_name]
        
        try:
            # Use default initial parameters if none provided
            if initial_params is None:
                initial_params = self._get_default_params(distro_name)
            
            # Define objective function
            def objective(params):
                try:
                    if distro_name == 'lognorm':
                        # Lognormal: params = [s, scale]
                        pdf_values = distribution.pdf(self._days, s=params[0], scale=params[1])
                    elif distro_name == 'weibull':
                        # Weibull: params = [c, scale]
                        pdf_values = distribution.pdf(self._days, c=params[0], scale=params[1])
                    elif distro_name == 'gamma':
                        # Gamma: params = [a, scale]
                        pdf_values = distribution.pdf(self._days, a=params[0], scale=params[1])
                    elif distro_name == 'beta':
                        # Beta: params = [a, b] (scaled to days)
                        pdf_values = distribution.pdf(self._days / self.max_los_days, 
                                                    a=params[0], b=params[1])
                    else:
                        # Other distributions with standard parameterization
                        pdf_values = distribution.pdf(self._days, *params)
                    
                    # Normalize
                    pdf_values = pdf_values / np.sum(pdf_values)
                    
                    # Calculate error
                    return mse(pdf_values[:len(target_data)], target_data)
                    
                except (ValueError, RuntimeWarning, FloatingPointError):
                    return np.inf
            
            # Optimization
            result = minimize(objective, initial_params, method='Nelder-Mead',
                            options={'maxiter': 1000})
            
            if result.success:
                # Calculate final error
                final_error = objective(result.x)
                return SingleFitResult(
                    success=True,
                    params=result.x.tolist(),
                    rel_train_error=final_error,
                    rel_test_error=final_error
                )
            else:
                return SingleFitResult(success=False)
                
        except Exception:
            return SingleFitResult(success=False)
    
    def _get_default_params(self, distro_name: str) -> List[float]:
        """Get reasonable default parameters for each distribution."""
        defaults = {
            'lognorm': [1.0, 10.0],
            'weibull': [2.0, 10.0],
            'normal': [10.0, 5.0],
            'exponential': [0.1],
            'gamma': [2.0, 5.0],
            'beta': [2.0, 5.0]
        }
        return defaults.get(distro_name, [1.0, 1.0])
    
    def generate_distribution_samples(self, distro_name: str, params: List[float], 
                                    n_days: int) -> np.ndarray:
        """
        Generate probability distribution for given parameters.
        
        Args:
            distro_name: Name of distribution
            params: Distribution parameters
            n_days: Number of days to generate
            
        Returns:
            Normalized probability distribution
        """
        if distro_name not in self.DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution: {distro_name}")
        
        distribution = self.DISTRIBUTIONS[distro_name]
        days = np.arange(1, n_days + 1)
        
        try:
            if distro_name == 'lognorm':
                pdf_values = distribution.pdf(days, s=params[0], scale=params[1])
            elif distro_name == 'weibull':
                pdf_values = distribution.pdf(days, c=params[0], scale=params[1])
            elif distro_name == 'gamma':
                pdf_values = distribution.pdf(days, a=params[0], scale=params[1])
            elif distro_name == 'beta':
                pdf_values = distribution.pdf(days / n_days, a=params[0], b=params[1])
            else:
                pdf_values = distribution.pdf(days, *params)
            
            # Normalize
            pdf_values = pdf_values / np.sum(pdf_values)
            return pdf_values
            
        except Exception:
            return np.zeros(n_days)
    
    def compare_distributions(self, target_data: np.ndarray, 
                            distributions: Optional[List[str]] = None) -> Dict[str, SingleFitResult]:
        """
        Compare multiple distributions against target data.
        
        Args:
            target_data: Target data to fit
            distributions: List of distribution names (if None, uses all)
            
        Returns:
            Dictionary of distribution names to fit results
        """
        if distributions is None:
            distributions = list(self.DISTRIBUTIONS.keys())
        
        results = {}
        for distro_name in distributions:
            results[distro_name] = self.fit_distribution(distro_name, target_data=target_data)
        
        return results
