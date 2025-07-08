"""Comparison utilities for fit results."""

import numpy as np
from typing import Dict, Any


def compare_fit_results(results1: Dict[str, Any], results2: Dict[str, Any], tolerance: float = 1e-4) -> bool:
    """Compare two sets of fit results for validation.
    
    Args:
        results1: First set of fit results
        results2: Second set of fit results
        tolerance: Tolerance for numerical comparison
        
    Returns:
        True if results match within tolerance
    """
    all_successful = True
    
    for distro in results2.keys():
        if distro not in results1:
            print(f"❌ Distribution {distro} not found in comparison results.")
            all_successful = False
            continue
            
        if distro == "compartmental":
            continue
            
        fit_result1 = results1[distro]
        fit_result2 = results2[distro]
        
        # Get mean errors for comparison
        train_error1 = getattr(fit_result1, 'train_relative_errors', [])
        train_error2 = getattr(fit_result2, 'train_relative_errors', [])
        test_error1 = getattr(fit_result1, 'test_relative_errors', [])
        test_error2 = getattr(fit_result2, 'test_relative_errors', [])
        
        if len(train_error1) > 0 and len(train_error2) > 0:
            train_diff = abs(np.nanmean(train_error1) - np.nanmean(train_error2))
        else:
            train_diff = 0
            
        if len(test_error1) > 0 and len(test_error2) > 0:
            test_diff = abs(np.nanmean(test_error1) - np.nanmean(test_error2))
        else:
            test_diff = 0
        
        if train_diff > tolerance or test_diff > tolerance:
            print(f"❌ Comparison failed for distribution: {distro}")
            print(f"Train Error Difference: {train_diff:.4f}")
            print(f"Test Error Difference: {test_diff:.4f}")
            all_successful = False
        else:
            print(f"✅ Comparison successful for distribution: {distro}")
    
    return all_successful
