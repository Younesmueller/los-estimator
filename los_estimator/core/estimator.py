"""
Main LOS Estimator class - the core interface for the package.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path

from .models import (
    EstimationParams, EstimationResult, SeriesData, 
    ErrorFunction, SingleFitResult, SeriesFitResult
)
from ..data.loader import DataLoader
from ..fitting.deconvolution import DeconvolutionEngine
from ..fitting.distributions import DistributionFitter
from ..visualization.plots import Visualizer
from ..utils.helpers import generate_run_name, setup_directories


class LOSEstimator:
    """
    Main class for Length of Stay estimation using deconvolution.
    
    This class provides a high-level interface for:
    - Loading and preprocessing data
    - Configuring estimation parameters
    - Running deconvolution estimation
    - Visualizing results
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize LOS Estimator.
        
        Args:
            data_dir: Base directory for data files
        """
        self.data_loader = DataLoader(data_dir)
        self.distribution_fitter = DistributionFitter()
        self.deconvolution_engine = DeconvolutionEngine(self.distribution_fitter)
        self.visualizer = Visualizer()
        
        # Storage for loaded data and results
        self.data: Optional[Dict] = None
        self.series_data: Optional[SeriesData] = None
        self.results: Optional[EstimationResult] = None
        self.params: Optional[EstimationParams] = None
    
    def load_data(self, los_file: str, incidence_file: str, icu_file: str,
                  init_params_file: str, mutants_file: str,
                  start_day: str = "2020-01-01", end_day: str = "2025-01-01") -> Dict:
        """
        Load all required data files.
        
        Args:
            los_file: Path to LOS distribution CSV file
            incidence_file: Path to incidence data CSV file  
            icu_file: Path to ICU occupancy CSV file
            init_params_file: Path to initial parameters CSV file
            mutants_file: Path to variant distribution Excel file
            start_day: Start date for analysis
            end_day: End date for analysis
            
        Returns:
            Dictionary containing loaded data
        """
        print("Loading data...")
        
        self.data = self.data_loader.load_complete_dataset(
            los_file=los_file,
            incidence_file=incidence_file,
            icu_file=icu_file,
            init_params_file=init_params_file,
            mutants_file=mutants_file,
            start_day=start_day,
            end_day=end_day
        )
        
        print(f"Loaded data from {self.data['df_occupancy'].index[0]} to {self.data['df_occupancy'].index[-1]}")
        print(f"Total days: {len(self.data['df_occupancy'])}")
        
        return self.data
    
    def configure_estimation(self, **kwargs) -> EstimationParams:
        """
        Configure estimation parameters.
        
        Args:
            **kwargs: Parameter values to override defaults
            
        Returns:
            EstimationParams object
        """
        self.params = EstimationParams(**kwargs)
        
        print("Estimation configuration:")
        print(f"  Kernel width: {self.params.kernel_width}")
        print(f"  Training width: {self.params.train_width}")
        print(f"  Test width: {self.params.test_width}")
        print(f"  Step size: {self.params.step}")
        print(f"  Fit admissions: {self.params.fit_admissions}")
        print(f"  Error function: {self.params.error_function.value}")
        
        return self.params
    
    def prepare_series_data(self, params: Optional[EstimationParams] = None) -> SeriesData:
        """
        Prepare time series data for estimation.
        
        Args:
            params: Estimation parameters (if None, uses configured params)
            
        Returns:
            SeriesData object
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data().")
        
        if params is None:
            if self.params is None:
                raise ValueError("Parameters must be configured first. Call configure_estimation().")
            params = self.params
        
        # Adjust series data creation to account for new_icu_date
        df_occupancy = self.data['df_occupancy'].copy()
        new_icu_date = self.data['new_icu_date']
        
        # Add new_icu_day to params temporarily
        new_icu_day = (new_icu_date - pd.Timestamp(params.start_day)).days
        
        self.series_data = SeriesData(df_occupancy, params)
        
        # Manually adjust windows to start from new_icu_day if fitting admissions
        if params.fit_admissions:
            from .models import WindowInfo
            start_window = new_icu_day + params.train_width
            self.series_data.windows = np.arange(
                start_window, 
                len(df_occupancy) - params.kernel_width, 
                params.step
            )
            self.series_data.window_infos = [
                WindowInfo(window, params) 
                for window in self.series_data.windows
            ]
            self.series_data.n_windows = len(self.series_data.windows)
        
        print(f"Prepared series data with {self.series_data.n_windows} windows")
        print(f"Window range: {self.series_data.windows[0]} to {self.series_data.windows[-1]}")
        
        return self.series_data
    
    def run_estimation(self, distributions: Optional[List[str]] = None,
                      params: Optional[EstimationParams] = None) -> EstimationResult:
        """
        Run the main LOS estimation.
        
        Args:
            distributions: List of distributions to fit (if None, uses default set)
            params: Estimation parameters (if None, uses configured params)
            
        Returns:
            EstimationResult object
        """
        if self.series_data is None:
            self.prepare_series_data(params)
        
        if params is None:
            params = self.params
        
        if distributions is None:
            distributions = ['lognorm', 'weibull', 'gamma', 'exponential']
        
        print(f"Running estimation with distributions: {distributions}")
        
        # Convert initial parameters dataframe to dictionary
        initial_params_dict = {}
        if self.data and 'df_init' in self.data:
            df_init = self.data['df_init']
            for distro in df_init.index:
                if distro in distributions:
                    initial_params_dict[distro] = df_init.loc[distro, 'params']
        
        # Run the estimation
        self.results = self.deconvolution_engine.fit_multiple_distributions(
            self.series_data, distributions, initial_params_dict, params
        )
        
        # Generate run name and set up directories
        run_name = generate_run_name(params)
        results_folder, figures_folder, animation_folder = setup_directories(run_name)
        
        # Store paths in results metadata
        self.results.metadata.update({
            'run_name': run_name,
            'results_folder': str(results_folder),
            'figures_folder': str(figures_folder),
            'animation_folder': str(animation_folder)
        })
        
        print(f"Estimation completed. Results saved to: {results_folder}")
        
        # Print summary
        summary = self.results.get_summary_stats()
        print("\nSummary:")
        print(summary.to_string(index=False))
        
        return self.results
    
    def visualize_results(self, save_plots: bool = True) -> Dict[str, str]:
        """
        Create visualizations of the estimation results.
        
        Args:
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary of plot names to file paths
        """
        if self.results is None:
            raise ValueError("No results to visualize. Run estimation first.")
        
        saved_plots = {}
        
        if save_plots and 'figures_folder' in self.results.metadata:
            figures_folder = self.results.metadata['figures_folder']
            saved_plots = self.visualizer.create_summary_report(
                self.results, figures_folder
            )
        
        # Create data overview plot
        if self.data:
            overview_fig = self.visualizer.plot_time_series_overview(
                self.data['df_occupancy'],
                new_icu_date=self.data['new_icu_date'],
                save_path=Path(figures_folder) / "time_series_overview.png" if save_plots else None
            )
            
            if save_plots:
                saved_plots['time_series_overview'] = str(Path(figures_folder) / "time_series_overview.png")
        
        if not save_plots:
            # Just show the plots
            self.visualizer.plot_distribution_comparison(self.results)
            self.visualizer.plot_fit_quality_heatmap(self.results)
        
        return saved_plots
    
    def get_best_distribution(self, metric: str = "test_error") -> str:
        """
        Get the name of the best performing distribution.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Name of best distribution
        """
        if self.results is None:
            raise ValueError("No results available. Run estimation first.")
        
        return self.results.get_best_distribution(metric)
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all fitted distributions.
        
        Returns:
            DataFrame with summary statistics
        """
        if self.results is None:
            raise ValueError("No results available. Run estimation first.")
        
        return self.results.get_summary_stats()
    
    def save_results(self, output_file: str) -> None:
        """
        Save estimation results to file.
        
        Args:
            output_file: Path to save results
        """
        if self.results is None:
            raise ValueError("No results to save. Run estimation first.")
        
        # Save summary statistics
        summary = self.get_summary_statistics()
        
        output_path = Path(output_file)
        if output_path.suffix == '.csv':
            summary.to_csv(output_file, index=False)
        elif output_path.suffix == '.xlsx':
            summary.to_excel(output_file, index=False)
        else:
            raise ValueError("Output file must be .csv or .xlsx")
        
        print(f"Results saved to: {output_file}")
    
    def run_complete_analysis(self, los_file: str, incidence_file: str, icu_file: str,
                            init_params_file: str, mutants_file: str,
                            distributions: Optional[List[str]] = None,
                            **estimation_params) -> EstimationResult:
        """
        Run complete analysis pipeline in one call.
        
        Args:
            los_file: Path to LOS distribution CSV file
            incidence_file: Path to incidence data CSV file
            icu_file: Path to ICU occupancy CSV file
            init_params_file: Path to initial parameters CSV file
            mutants_file: Path to variant distribution Excel file
            distributions: List of distributions to fit
            **estimation_params: Parameters for estimation
            
        Returns:
            EstimationResult object
        """
        print("=== Starting Complete LOS Estimation Analysis ===")
        
        # Load data
        self.load_data(
            los_file=los_file,
            incidence_file=incidence_file,
            icu_file=icu_file,
            init_params_file=init_params_file,
            mutants_file=mutants_file
        )
        
        # Configure estimation
        self.configure_estimation(**estimation_params)
        
        # Run estimation
        results = self.run_estimation(distributions)
        
        # Create visualizations
        saved_plots = self.visualize_results()
        
        print("=== Analysis Complete ===")
        print(f"Best distribution: {self.get_best_distribution()}")
        print(f"Plots saved: {list(saved_plots.keys())}")
        
        return results
