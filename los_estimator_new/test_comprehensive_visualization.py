"""
Test all the newly integrated visualization functions.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the package to path
sys.path.append(str(Path(__file__).parent))

from los_estimator import LOSEstimator, EstimationParams
from los_estimator.visualization.plots import Visualizer
from los_estimator.core.models import EstimationResult, SeriesFitResult, SingleFitResult, WindowInfo


def create_mock_series_data():
    """Create mock series data for testing."""
    n_days = 200
    n_windows = 10
    
    # Create mock data
    y_full = 1000 + 500 * np.sin(np.arange(n_days) * 0.02) + np.random.normal(0, 50, n_days)
    x_full = 50 + 30 * np.sin(np.arange(n_days) * 0.03) + np.random.normal(0, 10, n_days)
    
    # Create mock window infos
    params = EstimationParams(
        kernel_width=60, 
        step=7,
        train_width=50,
        test_width=14,
        los_cutoff=14
    )
    
    window_infos = []
    for i in range(n_windows):
        window_info = WindowInfo(window=i*10+50, params=params)
        window_infos.append(window_info)
    
    # Create a mock series data object
    class MockSeriesData:
        def __init__(self):
            self.y_full = y_full
            self.x_full = x_full
            self.window_infos = window_infos
            self.n_windows = n_windows
            self.n_days = n_days
    
    return MockSeriesData()


def test_new_visualization_functions():
    """Test all the newly integrated visualization functions."""
    print("=== Testing All New Visualization Functions ===")
    
    # Create mock estimation result
    params = EstimationParams(
        kernel_width=60, 
        step=7,
        train_width=50,
        test_width=14,
        los_cutoff=14
    )
    
    distros = ["gamma", "lognorm", "weibull"]
    
    # Create mock series results
    series_results = {}
    for distro in distros:
        series_result = SeriesFitResult(distro)
        
        # Add mock fit results with more realistic data
        for i in range(10):
            # Create mock kernel and curve data
            kernel = np.random.exponential(0.1, 100)
            kernel = kernel / np.sum(kernel)  # Normalize
            
            curve = 1000 + 200 * np.sin(np.arange(100) * 0.05) + np.random.normal(0, 20, 100)
            
            fit_result = SingleFitResult(
                success=True if i < 8 else False,  # 80% success rate
                params=[0.01 + np.random.normal(0, 0.002), 0.5, 2.0, 5.0],
                rel_train_error=0.1 + np.random.normal(0, 0.02),
                rel_test_error=0.12 + np.random.normal(0, 0.03)
            )
            
            # Add mock attributes for visualization
            fit_result.kernel = kernel
            fit_result.curve = curve
            
            window_info = WindowInfo(window=i+50, params=params)
            series_result.append(window_info, fit_result)
        
        series_result.bake()
        series_results[distro] = series_result
    
    estimation_result = EstimationResult(series_results, params)
    series_data = create_mock_series_data()
    
    # Test visualizer with all new functions
    visualizer = Visualizer()
    
    test_functions = [
        ("plot_error_failure_rate", lambda: visualizer.plot_error_failure_rate(
            estimation_result, "Mean Test Error", "Failure Rate Test")),
        
        ("plot_error_boxplots", lambda: visualizer.plot_error_boxplots(estimation_result)),
        
        ("plot_error_stripplot", lambda: visualizer.plot_error_stripplot(estimation_result)),
        
        ("plot_individual_distribution_analysis", lambda: visualizer.plot_individual_distribution_analysis(
            estimation_result, series_data, "gamma")),
        
        ("plot_all_predictions_combined", lambda: visualizer.plot_all_predictions_combined(
            estimation_result, series_data)),
        
        ("plot_all_errors_combined", lambda: visualizer.plot_all_errors_combined(estimation_result)),
        
        ("plot_distribution_kernels", lambda: visualizer.plot_distribution_kernels(
            estimation_result, np.random.exponential(0.1, 100))),
        
        ("create_summary_report", lambda: visualizer.create_summary_report(
            estimation_result, "test_summary_output", series_data=series_data))
    ]
    
    success_count = 0
    for func_name, func in test_functions:
        try:
            print(f"Testing {func_name}...")
            result = func()
            print(f"  ✓ {func_name} works")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {func_name} failed: {e}")
    
    print(f"\n=== Results: {success_count}/{len(test_functions)} functions work ===")
    
    if success_count == len(test_functions):
        print("🎉 All new visualization functions integrated successfully!")
        return True
    else:
        print("⚠️  Some functions need attention")
        return False


def test_comprehensive_integration():
    """Test that all functions are properly integrated into LOSEstimator."""
    print("\n=== Testing LOSEstimator Integration ===")
    
    # Check that all new methods are available
    visualizer = Visualizer()
    
    expected_methods = [
        'plot_error_failure_rate',
        'plot_error_boxplots', 
        'plot_error_stripplot',
        'plot_individual_distribution_analysis',
        'plot_all_predictions_combined',
        'plot_all_errors_combined',
        'plot_distribution_kernels'
    ]
    
    available_methods = []
    for method_name in expected_methods:
        if hasattr(visualizer, method_name):
            available_methods.append(method_name)
            print(f"  ✓ {method_name} available")
        else:
            print(f"  ✗ {method_name} missing")
    
    print(f"\nIntegration: {len(available_methods)}/{len(expected_methods)} methods available")
    
    return len(available_methods) == len(expected_methods)


def main():
    """Run all tests."""
    print("🚀 Testing Comprehensive Visualization Integration")
    print("=" * 60)
    
    success1 = test_new_visualization_functions()
    success2 = test_comprehensive_integration()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ ALL TESTS PASSED - Complete visualization integration successful!")
        print("\nNew visualization capabilities:")
        print("  • Error-failure rate scatter plots")
        print("  • Box plots and strip plots for errors")
        print("  • Individual distribution detailed analysis")
        print("  • Combined predictions and errors visualization")
        print("  • Distribution kernel analysis")
        print("  • Comprehensive summary reports")
        print("  • All functions integrated into Visualizer class")
        return True
    else:
        print("❌ Some tests failed - check the output above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
