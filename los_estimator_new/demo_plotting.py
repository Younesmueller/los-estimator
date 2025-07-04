"""
Demonstration of the integrated plotting functions.
"""

import sys
from pathlib import Path
import numpy as np

# Add the package to path
sys.path.append(str(Path(__file__).parent))

from los_estimator import LOSEstimator, EstimationParams
from los_estimator.data.loader import DataLoader


def demo_plotting_functions():
    """Demonstrate the integrated plotting functions with real data."""
    print("=== Plotting Functions Demo ===")
    
    # Load real data
    data_loader = DataLoader()
    try:
        # Try to load the Berlin ICU data
        data_path = Path("los-estimator/data/hosp_ag.csv")
        if data_path.exists():
            print(f"Loading data from {data_path}")
            df = data_loader.load_icu_data(str(data_path))
        else:
            print("Creating synthetic data for demo")
            # Create synthetic data
            dates = np.arange(np.datetime64('2020-03-01'), np.datetime64('2020-12-01'))
            icu_occupancy = 1000 + 500 * np.sin(np.arange(len(dates)) * 0.02) + np.random.normal(0, 50, len(dates))
            new_icu = np.maximum(0, 50 + 30 * np.sin(np.arange(len(dates)) * 0.03) + np.random.normal(0, 10, len(dates)))
            df = data_loader.create_synthetic_data(dates, icu_occupancy, new_icu)
    except Exception as e:
        print(f"Data loading failed: {e}")
        print("Creating synthetic data instead")
        dates = np.arange(np.datetime64('2020-03-01'), np.datetime64('2020-12-01'))
        icu_occupancy = 1000 + 500 * np.sin(np.arange(len(dates)) * 0.02) + np.random.normal(0, 50, len(dates))
        new_icu = np.maximum(0, 50 + 30 * np.sin(np.arange(len(dates)) * 0.03) + np.random.normal(0, 10, len(dates)))
        df = data_loader.create_synthetic_data(dates, icu_occupancy, new_icu)
    
    # Configure estimation parameters
    params = EstimationParams(
        distributions=["gamma", "lognormal", "weibull"],
        kernel_width=60,
        step=14,
        train_width=60,
        test_width=21,
        los_cutoff=14
    )
    
    # Create estimator
    estimator = LOSEstimator(params)
    
    # Run estimation (limited windows for demo)
    print("Running estimation...")
    try:
        results = estimator.fit(df, max_windows=5)  # Only 5 windows for quick demo
        
        # Demonstrate visualization functions
        print("\n=== Demonstrating Visualization Functions ===")
        
        # 1. Show successful fits
        print("1. Creating successful fits plot...")
        estimator.visualizer.plot_successful_fits(results, save_path="demo_successful_fits.png")
        print("   ✓ Saved: demo_successful_fits.png")
        
        # 2. Show distribution comparison
        print("2. Creating distribution comparison plot...")
        estimator.visualizer.plot_distribution_comparison(results, save_path="demo_distribution_comparison.png")
        print("   ✓ Saved: demo_distribution_comparison.png")
        
        # 3. Show fit quality heatmap
        print("3. Creating fit quality heatmap...")
        estimator.visualizer.plot_fit_quality_heatmap(results, save_path="demo_fit_quality_heatmap.png")
        print("   ✓ Saved: demo_fit_quality_heatmap.png")
        
        # 4. Show deconvolution visualization (for first window)
        print("4. Creating deconvolution visualization...")
        if len(results.series_results) > 0:
            try:
                estimator.visualizer.visualize_fit_deconvolution(
                    results, 
                    window_id=0,
                    save_path="demo_deconvolution.png"
                )
                print("   ✓ Saved: demo_deconvolution.png")
            except Exception as e:
                print(f"   ⚠ Deconvolution plot failed: {e}")
        
        # 5. Create comprehensive summary report
        print("5. Creating summary report...")
        saved_plots = estimator.visualizer.create_summary_report(results, "demo_plots")
        print("   ✓ Summary report created in 'demo_plots' folder:")
        for plot_name, path in saved_plots.items():
            print(f"     - {plot_name}: {path}")
        
        print("\n✅ All plotting functions demonstrated successfully!")
        print(f"Best distribution: {results.get_best_distribution()}")
        
    except Exception as e:
        print(f"Estimation failed: {e}")
        print("This is expected if the data format doesn't match expectations.")
        return False
    
    return True


if __name__ == "__main__":
    success = demo_plotting_functions()
    sys.exit(0 if success else 1)
