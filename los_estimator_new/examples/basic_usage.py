"""
Example usage of the LOS Estimator package.
"""
import sys
import os

# Add the parent directory to Python path to find los_estimator module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from los_estimator import LOSEstimator, EstimationParams
from los_estimator.utils.helpers import get_default_file_paths


def main():
    """Example of how to use the LOS Estimator package."""
    
    print("=== LOS Estimator Example ===")
    
    # 1. Initialize the estimator
    estimator = LOSEstimator()
    
    # 2. Get default file paths (you would replace these with your actual data files)
    file_paths = get_default_file_paths()
    
    print("Using example file paths:")
    for key, path in file_paths.items():
        print(f"  {key}: {path}")
    
    # For this example, let's assume we have the data files
    # In practice, you would set these to your actual file paths
    los_file = "path/to/los_distribution.csv"
    incidence_file = "path/to/incidence_data.csv" 
    icu_file = "path/to/icu_occupancy.csv"
    init_params_file = "path/to/initial_parameters.csv"
    mutants_file = "path/to/variant_data.xlsx"
    
    # 3. Option A: Run complete analysis in one call
    print("\n=== Option A: Complete Analysis ===")
    try:
        results = estimator.run_complete_analysis(
            los_file=los_file,
            incidence_file=incidence_file,
            icu_file=icu_file,
            init_params_file=init_params_file,
            mutants_file=mutants_file,
            # Custom parameters
            kernel_width=120,
            train_width=102,
            test_width=21,
            step=7,
            fit_admissions=True,
            error_function="mse",
            distributions=['lognorm', 'weibull', 'gamma']
        )
        
        print(f"Best distribution: {estimator.get_best_distribution()}")
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Could not run complete analysis (likely due to missing data files): {e}")
    
    # 3. Option B: Step-by-step analysis
    print("\n=== Option B: Step-by-Step Analysis ===")
    
    try:
        # Load data
        print("Loading data...")
        data = estimator.load_data(
            los_file=los_file,
            incidence_file=incidence_file,
            icu_file=icu_file,
            init_params_file=init_params_file,
            mutants_file=mutants_file
        )
        
        # Configure estimation parameters
        print("Configuring estimation...")
        params = estimator.configure_estimation(
            kernel_width=120,
            train_width=102,
            test_width=21,
            step=7,
            fit_admissions=True,
            smooth_data=False,
            error_function="mse"
        )
        
        # Prepare series data
        print("Preparing series data...")
        series_data = estimator.prepare_series_data()
        
        # Run estimation
        print("Running estimation...")
        results = estimator.run_estimation(['lognorm', 'weibull', 'gamma'])
        
        # Get results
        best_distro = estimator.get_best_distribution()
        summary_stats = estimator.get_summary_statistics()
        
        print(f"Best distribution: {best_distro}")
        print("\nSummary statistics:")
        print(summary_stats)
        
        # Visualize results
        print("Creating visualizations...")
        saved_plots = estimator.visualize_results()
        print(f"Saved plots: {list(saved_plots.keys())}")
        
        # Save results
        estimator.save_results("example_results.csv")
        
    except Exception as e:
        print(f"Could not run step-by-step analysis (likely due to missing data files): {e}")
    
    # 4. Working with estimation parameters
    print("\n=== Parameter Configuration Examples ===")
    
    # Create parameters with defaults
    params1 = EstimationParams()
    print(f"Default parameters: kernel_width={params1.kernel_width}, step={params1.step}")
    
    # Create parameters with custom values
    params2 = EstimationParams(
        kernel_width=180,
        train_width=150,
        test_width=28,
        step=14,
        fit_admissions=False,
        error_function="weighted_mse"
    )
    print(f"Custom parameters: kernel_width={params2.kernel_width}, step={params2.step}")
    
    # 5. Using individual components
    print("\n=== Using Individual Components ===")
    
    from los_estimator.data.loader import DataLoader
    from los_estimator.fitting.distributions import DistributionFitter
    from los_estimator.visualization.plots import Visualizer
    
    # Data loader example
    data_loader = DataLoader()
    print("DataLoader initialized")
    
    # Distribution fitter example
    dist_fitter = DistributionFitter()
    print(f"Available distributions: {list(dist_fitter.DISTRIBUTIONS.keys())}")
    
    # Visualizer example
    visualizer = Visualizer()
    print("Visualizer initialized")
    
    print("\n=== Example Complete ===")
    print("This example shows how to use the LOS Estimator package.")
    print("Replace the file paths with your actual data files to run real analysis.")


if __name__ == "__main__":
    main()
