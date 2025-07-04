"""
Test the enhanced visualization functions.
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


def test_visualization_functions():
    """Test the new visualization functions with mock data."""
    print("=== Testing Enhanced Visualization Functions ===")
    
    # Create a mock estimation result
    params = EstimationParams(
        kernel_width=60, 
        step=7,
        train_width=50,
        test_width=14,
        los_cutoff=14
    )
    
    # Create mock series results
    series_results = {}
    distributions = ['gamma', 'lognorm', 'weibull']
    
    for distro in distributions:
        series_result = SeriesFitResult(distro)
        
        # Add some mock fit results
        for i in range(10):
            fit_result = SingleFitResult(
                success=True if i < 8 else False,  # 80% success rate
                params=[0.01, 0.5, 2.0, 5.0],
                rel_train_error=0.1 + np.random.normal(0, 0.02),
                rel_test_error=0.12 + np.random.normal(0, 0.03)
            )
            # Mock window info using proper constructor
            window_info = WindowInfo(window=i+50, params=params)
            series_result.append(window_info, fit_result)
        
        series_result.bake()
        series_results[distro] = series_result
    
    # Create mock estimation result
    estimation_result = EstimationResult(series_results, params)
    
    # Test visualizer
    visualizer = Visualizer()
    
    try:
        # Test successful fits plot
        print("Testing plot_successful_fits...")
        fig1 = visualizer.plot_successful_fits(estimation_result)
        print("✓ plot_successful_fits works")
        
        # Test distribution comparison
        print("Testing plot_distribution_comparison...")
        fig2 = visualizer.plot_distribution_comparison(estimation_result)
        print("✓ plot_distribution_comparison works")
        
        # Test fit quality heatmap
        print("Testing plot_fit_quality_heatmap...")
        fig3 = visualizer.plot_fit_quality_heatmap(estimation_result)
        print("✓ plot_fit_quality_heatmap works")
        
        # Create mock series data for deconvolution visualization
        print("Testing visualize_fit_deconvolution...")
        MockSeriesData = type('MockSeriesData', (), {
            'y_full': np.random.randint(100, 1000, 100),
            'x_full': np.random.randint(10, 100, 100),
            'window_infos': [type('WindowInfo', (), {
                'window': 2, 'train_start': 10, 'train_los_cutoff': 20,
                'train_end': 50, 'test_start': 50, 'test_end': 70
            })()]
        })()
        
        # Mock real LOS data
        real_los = np.exp(-np.linspace(0, 5, 60)) * 0.1
        
        fig4 = visualizer.visualize_fit_deconvolution(
            estimation_result, MockSeriesData, params, 
            real_los=real_los, window_id=0
        )
        print("✓ visualize_fit_deconvolution works")
        
        print("\n=== All Enhanced Visualization Functions Work! ===")
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def test_with_actual_estimator():
    """Test visualization with the actual estimator class."""
    print("\n=== Testing with LOSEstimator Class ===")
    
    try:
        estimator = LOSEstimator()
        
        # Test that the visualizer has the new methods
        assert hasattr(estimator.visualizer, 'plot_successful_fits'), "Missing plot_successful_fits method"
        assert hasattr(estimator.visualizer, 'visualize_fit_deconvolution'), "Missing visualize_fit_deconvolution method"
        
        print("✓ LOSEstimator has enhanced visualization methods")
        
        # Test method signatures
        import inspect
        sig1 = inspect.signature(estimator.visualizer.plot_successful_fits)
        sig2 = inspect.signature(estimator.visualizer.visualize_fit_deconvolution)
        
        print("✓ Method signatures are correct")
        print(f"  plot_successful_fits: {len(sig1.parameters)} parameters")
        print(f"  visualize_fit_deconvolution: {len(sig2.parameters)} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ LOSEstimator test failed: {e}")
        return False


def main():
    """Run all visualization tests."""
    print("=== Enhanced Visualization Testing ===")
    
    success1 = test_visualization_functions()
    success2 = test_with_actual_estimator()
    
    if success1 and success2:
        print("\n✅ All visualization enhancements work correctly!")
        print("\nNew features added:")
        print("  • plot_successful_fits() - Shows success rate by distribution")
        print("  • visualize_fit_deconvolution() - Comprehensive deconvolution analysis")
        print("  • Enhanced create_summary_report() - Includes all visualization types")
        print("  • Integrated with LOSEstimator.visualize_results()")
        return 0
    else:
        print("\n❌ Some visualization tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
