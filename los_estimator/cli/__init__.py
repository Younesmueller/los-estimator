"""Command-line interface for LOS Estimator."""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

from ..core import Params
from ..data import DataLoader
from ..config import DataConfig, ModelConfig, OutputConfig
from ..fitting import MultiSeriesFitter
from ..visualization import DeconvolutionPlots, DeconvolutionAnimator


def create_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Length of Stay Estimator for ICU data using deconvolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  los-estimator --los-file data/los.csv --init-params data/params.csv
  
  # Run with custom parameters
  los-estimator --los-file data/los.csv --init-params data/params.csv --kernel-width 180 --step 14
        """
    )
    
    # Required file arguments
    parser.add_argument('--los-file', required=True, help='Path to LOS distribution CSV file')
    parser.add_argument('--init-params', required=True, help='Path to initial parameters CSV file')
    
    # Optional file arguments
    parser.add_argument('--mutants-file', help='Path to variant distribution Excel file')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    
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
                       help='Apply smoothing to input data')
    
    # Output options
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--no-animations', action='store_true', help='Skip animation generation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return parser


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    # Check file existence
    if not Path(args.los_file).exists():
        errors.append(f"LOS file not found: {args.los_file}")
    if not Path(args.init_params).exists():
        errors.append(f"Initial parameters file not found: {args.init_params}")
    if args.mutants_file and not Path(args.mutants_file).exists():
        errors.append(f"Mutants file not found: {args.mutants_file}")
    
    # Check date format
    try:
        pd.Timestamp(args.start_day)
        pd.Timestamp(args.end_day)
    except ValueError as e:
        errors.append(f"Invalid date format: {e}")
    
    # Check parameter ranges
    if args.kernel_width <= 0:
        errors.append("Kernel width must be positive")
    if args.los_cutoff <= 0:
        errors.append("LOS cutoff must be positive")
    if args.train_width <= 0:
        errors.append("Train width must be positive")
    if args.test_width <= 0:
        errors.append("Test width must be positive")
    if args.step <= 0:
        errors.append("Step size must be positive")
    
    return errors


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    errors = validate_arguments(args)
    if errors:
        print("Error: Invalid arguments:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    try:
        # Configure data loading
        data_config = DataConfig(
            los_file=args.los_file,
            init_params_file=args.init_params,
            mutants_file=args.mutants_file,
            start_day=args.start_day,
            end_day=args.end_day
        )
        
        # Configure model parameters
        model_config = ModelConfig(
            kernel_width=args.kernel_width,
            los_cutoff=args.los_cutoff,
            train_width=args.train_width,
            test_width=args.test_width,
            step=args.step,
            fit_admissions=args.fit_admissions,
            smooth_data=args.smooth_data
        )
        
        # Configure output
        output_config = OutputConfig(
            output_dir=args.output_dir,
            save_plots=not args.no_plots,
            save_animations=not args.no_animations,
            verbose=args.verbose
        )
        
        if args.verbose:
            print("Starting LOS estimation with configuration:")
            print(f"  Data config: {data_config}")
            print(f"  Model config: {model_config}")
            print(f"  Output config: {output_config}")
        
        # Load data
        data_loader = DataLoader(data_config)
        data = data_loader.load_all_data()
        
        if args.verbose:
            print("Data loaded successfully")
        
        # Create parameters object
        params = Params(
            kernel_width=args.kernel_width,
            los_cutoff=args.los_cutoff,
            train_width=args.train_width,
            test_width=args.test_width,
            step=args.step,
            fit_admissions=args.fit_admissions,
            smooth_data=args.smooth_data
        )
        
        # Run fitting
        fitter = MultiSeriesFitter(params)
        results = fitter.fit_all_series(data)
        
        if args.verbose:
            print(f"Fitting completed. {len(results)} series fitted.")
        
        # Generate outputs
        if not args.no_plots:
            plotter = DeconvolutionPlots()
            plotter.create_all_plots(results, output_config.output_dir)
            if args.verbose:
                print("Plots saved")
        
        if not args.no_animations:
            animator = DeconvolutionAnimator()
            animator.create_animations(results, output_config.output_dir)
            if args.verbose:
                print("Animations saved")
        
        print("LOS estimation completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
