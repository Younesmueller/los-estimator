"""
Visualization utilities for LOS estimation results.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from ..core.models import EstimationResult, SeriesFitResult, EstimationParams


class Visualizer:
    """Visualization utilities for LOS estimation results."""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("default")
        
        self.figsize = figsize
        self.colors = self._get_color_palette()
        
        # Set high-quality defaults
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 100
    
    def _get_color_palette(self) -> List[str]:
        """Get extended color palette."""
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors += ["#FFA07A", "#20B2AA", "#FF6347", "#808000", "#FF00FF", 
                  "#FFD700", "#00FF00", "#00FFFF", "#0000FF", "#8A2BE2"]
        return colors
    
    def plot_time_series_overview(self, df_occupancy: pd.DataFrame, 
                                new_icu_date: Optional[pd.Timestamp] = None,
                                sentinel_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot overview of time series data (incidences and ICU occupancy).
        
        Args:
            df_occupancy: DataFrame with occupancy data
            new_icu_date: Date when ICU reporting started
            sentinel_range: Tuple of (start, end) for sentinel period
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Plot ICU occupancy
        df_occupancy["icu"].plot(ax=axes[0], label="ICU Occupancy", color=self.colors[0])
        axes[0].set_title("ICU Bed Occupancy")
        axes[0].set_ylabel("Number of Beds")
        axes[0].legend()
        
        # Plot new ICU admissions
        df_occupancy["new_icu"].plot(ax=axes[1], label="New ICU (raw)", color=self.colors[1], alpha=0.7)
        if "new_icu_smooth" in df_occupancy.columns:
            df_occupancy["new_icu_smooth"].plot(ax=axes[1], label="New ICU (7-day avg)", 
                                              color=self.colors[2], linewidth=2)
        axes[1].set_title("Daily New ICU Admissions")
        axes[1].set_ylabel("Number of Admissions")
        axes[1].legend()
        
        # Add vertical lines for important dates
        if new_icu_date:
            for ax in axes:
                ax.axvline(new_icu_date, color="black", linestyle="--", 
                          label="First ICU Report", alpha=0.7)
        
        # Add sentinel period highlighting
        if sentinel_range:
            for ax in axes:
                ax.axvspan(sentinel_range[0], sentinel_range[1], 
                          color="green", alpha=0.1, label="Sentinel Period")
        
        # Format x-axis
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_distribution_comparison(self, estimation_result: EstimationResult,
                                   metric: str = "test_error",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of different distributions.
        
        Args:
            estimation_result: Results from estimation
            metric: Metric to compare ('test_error', 'train_error', 'success_rate')
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Get summary statistics
        summary_stats = estimation_result.get_summary_stats()
        
        # 1. Error comparison
        if metric in ["test_error", "train_error"]:
            error_col = f"mean_{metric}"
            std_col = f"std_{metric}"
            
            x_pos = np.arange(len(summary_stats))
            axes[0].bar(x_pos, summary_stats[error_col], 
                       yerr=summary_stats[std_col], 
                       color=self.colors[:len(summary_stats)], alpha=0.7)
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(summary_stats['distribution'], rotation=45)
            axes[0].set_ylabel(f"Mean {metric.replace('_', ' ').title()}")
            axes[0].set_title(f"Distribution Comparison - {metric.replace('_', ' ').title()}")
        
        # 2. Success rate comparison
        x_pos = np.arange(len(summary_stats))
        axes[1].bar(x_pos, summary_stats['success_rate'], 
                   color=self.colors[:len(summary_stats)], alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(summary_stats['distribution'], rotation=45)
        axes[1].set_ylabel("Success Rate")
        axes[1].set_title("Optimization Success Rate by Distribution")
        axes[1].set_ylim(0, 1)
        
        # 3. Error over time for best distribution
        best_distro = estimation_result.get_best_distribution(metric)
        best_result = estimation_result.series_results[best_distro]
        
        window_ids = np.arange(len(best_result.fit_results))
        axes[2].plot(window_ids, best_result.train_relative_errors, 
                    label="Train Error", color=self.colors[0])
        axes[2].plot(window_ids, best_result.test_relative_errors, 
                    label="Test Error", color=self.colors[1])
        axes[2].set_xlabel("Window ID")
        axes[2].set_ylabel("Relative Error")
        axes[2].set_title(f"Error Over Time - {best_distro}")
        axes[2].legend()
        
        # 4. Parameter evolution for best distribution
        if hasattr(best_result, 'transition_rates'):
            valid_windows = ~np.isnan(best_result.transition_rates)
            axes[3].plot(window_ids[valid_windows], 
                        best_result.transition_rates[valid_windows],
                        label="Transition Rate", color=self.colors[0], marker='o', alpha=0.7)
            
            if hasattr(best_result, 'transition_delays'):
                valid_delays = ~np.isnan(best_result.transition_delays)
                ax3_twin = axes[3].twinx()
                ax3_twin.plot(window_ids[valid_delays], 
                            best_result.transition_delays[valid_delays],
                            label="Transition Delay", color=self.colors[1], marker='s', alpha=0.7)
                ax3_twin.set_ylabel("Transition Delay")
                ax3_twin.legend(loc='upper right')
            
            axes[3].set_xlabel("Window ID")
            axes[3].set_ylabel("Transition Rate")
            axes[3].set_title(f"Parameter Evolution - {best_distro}")
            axes[3].legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_fit_quality_heatmap(self, estimation_result: EstimationResult,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot heatmap of fit quality across distributions and windows.
        
        Args:
            estimation_result: Results from estimation
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        # Prepare data for heatmap
        distros = estimation_result.distros
        n_windows = len(estimation_result.series_results[distros[0]].fit_results)
        
        # Create matrices for test errors and success rates
        test_errors = np.full((len(distros), n_windows), np.nan)
        success_matrix = np.zeros((len(distros), n_windows))
        
        for i, distro in enumerate(distros):
            result = estimation_result.series_results[distro]
            test_errors[i, :] = result.test_relative_errors
            success_matrix[i, :] = [1 if fr and fr.success else 0 for fr in result.fit_results]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot test errors heatmap
        im1 = axes[0].imshow(test_errors, cmap='viridis_r', aspect='auto')
        axes[0].set_yticks(range(len(distros)))
        axes[0].set_yticklabels(distros)
        axes[0].set_xlabel("Window ID")
        axes[0].set_title("Test Error by Distribution and Window")
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label("Test Error")
        
        # Plot success rate heatmap
        im2 = axes[1].imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1].set_yticks(range(len(distros)))
        axes[1].set_yticklabels(distros)
        axes[1].set_xlabel("Window ID")
        axes[1].set_title("Optimization Success by Distribution and Window")
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label("Success (1) / Failure (0)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_vs_actual(self, series_data, estimation_result: EstimationResult,
                                distro_name: Optional[str] = None,
                                window_range: Optional[Tuple[int, int]] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot predicted vs actual values for a specific distribution.
        
        Args:
            series_data: SeriesData object with original data
            estimation_result: Results from estimation
            distro_name: Distribution to plot (if None, uses best)
            window_range: Range of windows to plot (start, end)
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        if distro_name is None:
            distro_name = estimation_result.get_best_distribution()
        
        if window_range is None:
            window_range = (0, min(20, len(series_data)))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot actual data
        ax.plot(series_data.y_full, label="Actual ICU Occupancy", 
               color=self.colors[0], alpha=0.7)
        
        # This would require implementing the prediction generation
        # which would need the actual convolution models
        ax.set_xlabel("Days")
        ax.set_ylabel("ICU Occupancy")
        ax.set_title(f"Prediction vs Actual - {distro_name}")
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def create_summary_report(self, estimation_result: EstimationResult,
                            output_dir: str) -> Dict[str, str]:
        """
        Create a comprehensive summary report with multiple plots.
        
        Args:
            estimation_result: Results from estimation
            output_dir: Directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        # Distribution comparison
        comp_path = output_path / "distribution_comparison.png"
        self.plot_distribution_comparison(estimation_result, save_path=str(comp_path))
        saved_plots["distribution_comparison"] = str(comp_path)
        
        # Fit quality heatmap
        heatmap_path = output_path / "fit_quality_heatmap.png"
        self.plot_fit_quality_heatmap(estimation_result, save_path=str(heatmap_path))
        saved_plots["fit_quality_heatmap"] = str(heatmap_path)
        
        # Summary statistics table
        summary_stats = estimation_result.get_summary_stats()
        stats_path = output_path / "summary_statistics.csv"
        summary_stats.to_csv(stats_path, index=False)
        saved_plots["summary_statistics"] = str(stats_path)
        
        return saved_plots
