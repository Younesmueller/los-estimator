"""
Command-line interface for LOS Estimator.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .core.estimator import LOSEstimator
from .utils.helpers import get_default_file_paths, validate_data_files


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Length of Stay Estimator for ICU data using deconvolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  los-estimator --los-file data/los.csv --incidence-file data/cases.csv --icu-file data/icu.csv --init-params data/params.csv --mutants-file data/mutants.xlsx

  # Run with custom parameters
  los-estimator --los-file data/los.csv --incidence-file data/cases.csv --icu-file data/icu.csv --init-params data/params.csv --mutants-file data/mutants.xlsx --kernel-width 180 --step 14
        """
    )
    
    # Required file arguments
    parser.add_argument('--los-file', required=True, help='Path to LOS distribution CSV file')
    parser.add_argument('--incidence-file', required=True, help='Path to incidence data CSV file')
    parser.add_argument('--icu-file', required=True, help='Path to ICU occupancy CSV file')
    parser.add_argument('--init-params', required=True, help='Path to initial parameters CSV file')
    parser.add_argument('--mutants-file', required=True, help='Path to variant distribution Excel file')
    
    # Optional date arguments
    parser.add_argument('--start-day', default='2020-01-01', help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-day', default='2025-01-01', help='End date for analysis (YYYY-MM-DD)')
    
    # Estimation parameters
    parser.add_argument('--kernel-width', type=int, default=120, help='Kernel width for deconvolution')
    parser.add_argument('--los-cutoff', type=int, default=60, help='LOS cutoff in days')
    parser.add_argument('--train-width', type=int, default=102, help='Training window width')
    parser.add_argument('--test-width', type=int, default=21, help='Test window width')
    parser.add_argument('--step', type=int, default=7, help='Step size between windows')
    
    # Data processing options
    parser.add_argument('--fit-admissions', action='store_true', default=True, 
                       help='Fit to admissions data (default: True)')
    parser.add_argument('--fit-incidence', dest='fit_admissions', action='store_false',
                       help='Fit to incidence data instead of admissions')
    parser.add_argument('--smooth-data', action='store_true', default=False,
                       help='Use smoothed data')
    
    # Optimization options
    parser.add_argument('--error-function', choices=['mse', 'weighted_mse', 'mae'], 
                       default='mse', help='Error function for optimization')
    parser.add_argument('--no-reuse-params', dest='reuse_last_parametrization', 
                       action='store_false', default=True,
                       help='Do not reuse last parametrization')
    parser.add_argument('--no-variable-kernels', dest='variable_kernels', 
                       action='store_false', default=True,
                       help='Do not use variable kernels')
    
    # Distribution selection
    parser.add_argument('--distributions', nargs='+', 
                       choices=['lognorm', 'weibull', 'gamma', 'exponential', 'normal', 'beta'],
                       default=['lognorm', 'weibull', 'gamma', 'exponential'],
                       help='Distributions to fit')
    
    # Output options
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--validate-only', action='store_true', help='Only validate input files')
    
    args = parser.parse_args()
    
    # Validate input files
    file_paths = {
        'LOS file': args.los_file,
        'Incidence file': args.incidence_file,
        'ICU file': args.icu_file,
        'Initial parameters': args.init_params,
        'Mutants file': args.mutants_file
    }
    
    validation_results = validate_data_files(file_paths)
    
    print("=== File Validation ===")
    for description, result in validation_results.items():
        status = "✓" if result['exists'] and result['is_file'] else "✗"
        size = f"({result['size_mb']:.1f} MB)" if result['size_mb'] > 0 else ""
        print(f"{status} {description}: {result['path']} {size}")
    
    # Check if any files are missing
    missing_files = [desc for desc, result in validation_results.items() 
                    if not (result['exists'] and result['is_file'])]
    
    if missing_files:
        print(f"\nError: Missing or invalid files: {', '.join(missing_files)}")
        return 1
    
    if args.validate_only:
        print("\nValidation complete. All files are accessible.")
        return 0
    
    # Initialize estimator
    print("\n=== Initializing LOS Estimator ===")
    estimator = LOSEstimator()
    
    try:
        # Run complete analysis
        results = estimator.run_complete_analysis(
            los_file=args.los_file,
            incidence_file=args.incidence_file,
            icu_file=args.icu_file,
            init_params_file=args.init_params,
            mutants_file=args.mutants_file,
            distributions=args.distributions,
            # Estimation parameters
            kernel_width=args.kernel_width,
            los_cutoff=args.los_cutoff,
            train_width=args.train_width,
            test_width=args.test_width,
            step=args.step,
            fit_admissions=args.fit_admissions,
            smooth_data=args.smooth_data,
            error_function=args.error_function,
            reuse_last_parametrization=args.reuse_last_parametrization,
            variable_kernels=args.variable_kernels,
            start_day=args.start_day,
            end_day=args.end_day
        )
        
        # Save results
        if 'results_folder' in results.metadata:
            results_file = Path(results.metadata['results_folder']) / 'summary_results.csv'
            estimator.save_results(str(results_file))
        
        print(f"\n=== Analysis Complete ===")
        print(f"Results saved to: {results.metadata.get('results_folder', 'results')}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
