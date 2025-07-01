"""
Integration example that works with the existing los-estimator data structure.
This shows how to use the new package with your existing data.
"""

import sys
from pathlib import Path

# Add the package to path
sys.path.append(str(Path(__file__).parent.parent))

from los_estimator import LOSEstimator, EstimationParams


def run_with_existing_data():
    """Example using the existing data structure."""
    
    print("=== LOS Estimator - Integration with Existing Data ===")
    
    # Define paths to existing data files
    base_dir = Path("los-estimator")  # Adjust this path as needed
    
    data_files = {
        'los_file': base_dir / "01_create_los_profiles" / "berlin" / "output_los" / "los_berlin_all.csv",
        'incidence_file': base_dir / "data" / "cases.csv", 
        'icu_file': base_dir / "data" / "Intensivregister_Bundeslaender_Kapazitaeten.csv",
        'init_params_file': base_dir / "02_fit_los_distributions" / "output_los" / "los_berlin_all" / "fit_results.csv",
        'mutants_file': base_dir / "data" / "VOC_VOI_Tabelle.xlsx"
    }
    
    # Check which files exist
    print("Checking data files:")
    missing_files = []
    for name, path in data_files.items():
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} (missing)")
            missing_files.append(name)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        print("This example shows the API usage. Replace paths with your actual data files.")
        show_api_example()
        return
    
    # Initialize estimator
    estimator = LOSEstimator()
    
    try:
        print("\n=== Running Analysis ===")
        
        # Run the complete analysis
        results = estimator.run_complete_analysis(
            los_file=str(data_files['los_file']),
            incidence_file=str(data_files['incidence_file']),
            icu_file=str(data_files['icu_file']),
            init_params_file=str(data_files['init_params_file']),
            mutants_file=str(data_files['mutants_file']),
            # Parameters matching original analysis
            kernel_width=120,
            train_width=102,
            test_width=21,
            step=7,
            fit_admissions=True,
            smooth_data=False,
            error_function="mse",
            reuse_last_parametrization=True,
            variable_kernels=True,
            distributions=['lognorm', 'weibull', 'gamma', 'exponential']
        )
        
        print(f"\n=== Results ===")
        print(f"Best distribution: {estimator.get_best_distribution()}")
        
        # Print summary statistics
        summary = estimator.get_summary_statistics()
        print(f"\nSummary Statistics:")
        print(summary.to_string(index=False))
        
        # Results are automatically saved to the results folder
        print(f"\nResults saved to: {results.metadata['results_folder']}")
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        print("This might be due to data format differences.")
        show_api_example()


def show_api_example():
    """Show API usage example without requiring actual data files."""
    
    print("\n=== API Usage Example ===")
    
    # Initialize estimator
    estimator = LOSEstimator()
    
    # Create estimation parameters
    params = EstimationParams(
        kernel_width=120,
        train_width=102,
        test_width=21,
        step=7,
        fit_admissions=True,
        smooth_data=False,
        error_function="mse"
    )
    
    print(f"Created parameters: kernel_width={params.kernel_width}, step={params.step}")
    
    # Show step-by-step process (without actually running it)
    print("\nStep-by-step process:")
    print("1. estimator.load_data(...)")
    print("2. estimator.configure_estimation(...)")
    print("3. estimator.prepare_series_data()")
    print("4. estimator.run_estimation(['lognorm', 'weibull', 'gamma'])")
    print("5. estimator.visualize_results()")
    print("6. estimator.get_best_distribution()")
    print("7. estimator.save_results('results.csv')")
    
    # Show individual component usage
    print("\n=== Individual Components ===")
    
    from los_estimator.fitting.distributions import DistributionFitter
    from los_estimator.data.loader import DataLoader
    from los_estimator.visualization.plots import Visualizer
    from los_estimator.utils.helpers import generate_run_name
    
    # Distribution fitter
    fitter = DistributionFitter()
    print(f"Available distributions: {list(fitter.DISTRIBUTIONS.keys())}")
    
    # Data loader
    loader = DataLoader()
    print("DataLoader initialized")
    
    # Visualizer
    viz = Visualizer()
    print("Visualizer initialized")
    
    # Generate run name
    run_name = generate_run_name(params)
    print(f"Generated run name: {run_name}")


def compare_with_original():
    """Compare the new package structure with the original fit_deconvolution.py approach."""
    
    print("\n=== Comparison: New Package vs Original Script ===")
    
    print("\nOriginal approach (fit_deconvolution.py):")
    print("- Single monolithic script")
    print("- Hard-coded parameters and file paths")
    print("- Mixed data loading, processing, and visualization")
    print("- Difficult to reuse components")
    print("- No clear API for external use")
    
    print("\nNew package approach (los_estimator):")
    print("- Modular structure with clear separation of concerns")
    print("- Configurable parameters via EstimationParams")
    print("- Reusable components (DataLoader, DistributionFitter, etc.)")
    print("- Clean API for both programmatic and CLI use")
    print("- Easy to extend with new distributions or error functions")
    print("- Professional package structure with tests and documentation")
    
    print("\nKey benefits:")
    print("- Easier maintenance and debugging")
    print("- Better testability")
    print("- Reusable for different datasets")
    print("- Can be installed and imported like any Python package")
    print("- Clear documentation and examples")


if __name__ == "__main__":
    run_with_existing_data()
    compare_with_original()
