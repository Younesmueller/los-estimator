"""
Visualization utilities for LOS estimation results.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import os
import shutil
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
                            output_dir: str, series_data=None, 
                            real_los: Optional[np.ndarray] = None,
                            df_mutant_selection: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        Create a comprehensive summary report with multiple plots.
        
        Args:
            estimation_result: Results from estimation
            output_dir: Directory to save plots
            series_data: Series data for additional plots
            real_los: Real LOS distribution for comparison
            df_mutant_selection: Variant/mutation data
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        # Distribution comparison
        comp_path = output_path / "distribution_comparison.png"
        f = self.plot_distribution_comparison(estimation_result, save_path=str(comp_path))
        saved_plots["distribution_comparison"] = str(comp_path)
        f.show()

        # Fit quality heatmap
        heatmap_path = output_path / "fit_quality_heatmap.png"
        f = self.plot_fit_quality_heatmap(estimation_result, save_path=str(heatmap_path))
        saved_plots["fit_quality_heatmap"] = str(heatmap_path)
        f.show()

        # Successful fits plot
        success_path = output_path / "successful_fits.png"
        f = self.plot_successful_fits(estimation_result, save_path=str(success_path))
        saved_plots["successful_fits"] = str(success_path)
        f.show()

        # Error boxplots
        boxplot_path = output_path / "error_boxplots.png"
        f = self.plot_error_boxplots(estimation_result, save_path=str(boxplot_path))
        saved_plots["error_boxplots"] = str(boxplot_path)
        f.show()

        # Error stripplot
        stripplot_path = output_path / "error_stripplot.png"
        f = self.plot_error_stripplot(estimation_result, save_path=str(stripplot_path))
        saved_plots["error_stripplot"] = str(stripplot_path)
        f.show()

        # Error-failure rate plots
        error_failure_path = output_path / "error_failure_rates.png"
        f = self.plot_error_failure_rate(estimation_result, "Mean Test Error", "Failure Rate Test", 
                                   save_path=str(error_failure_path))
        saved_plots["error_failure_rates"] = str(error_failure_path)
        f.show()

        # All errors combined
        all_errors_path = output_path / "all_errors_combined.png"
        f = self.plot_all_errors_combined(estimation_result, save_path=str(all_errors_path))
        saved_plots["all_errors_combined"] = str(all_errors_path)
        f.show()
        
        # Individual distribution analyses
        for distro_name in estimation_result.distros:
            individual_path = output_path / f"individual_analysis_{distro_name}.png"
            f = self.plot_individual_distribution_analysis(
                estimation_result, series_data, distro_name, 
                save_path=str(individual_path)
            )
            saved_plots[f"individual_analysis_{distro_name}"] = str(individual_path)
            f.show()
        
        # Distribution kernels
        if real_los is not None:
            f = kernel_figures = self.plot_distribution_kernels(
                estimation_result, real_los, save_path=str(output_path / "kernels")
            )
            saved_plots.update({f"kernel_{k}": str(output_path / f"kernels_{k}_kernels.png") 
                              for k in kernel_figures.keys()})
            f.show()
        
        # All predictions combined
        if series_data is not None:
            combined_path = output_path / "all_predictions_combined.png"
            f = self.plot_all_predictions_combined(estimation_result, series_data, 
                                             save_path=str(combined_path))
            saved_plots["all_predictions_combined"] = str(combined_path)
            f.show()
        
        # Deconvolution visualization (if series_data is available)
        if series_data is not None:
            deconv_path = output_path / "deconvolution_analysis.png"
            f = self.visualize_fit_deconvolution(
                estimation_result, series_data, estimation_result.params,
                real_los=real_los, df_mutant_selection=df_mutant_selection,
                save_path=str(deconv_path)
            )
            saved_plots["deconvolution_analysis"] = str(deconv_path)
            f.show()
        
        # Summary statistics table
        summary_stats = estimation_result.get_summary_stats()
        stats_path = output_path / "summary_statistics.csv"
        summary_stats.to_csv(stats_path, index=False)
        saved_plots["summary_statistics"] = str(stats_path)
        
        return saved_plots
    
    def plot_successful_fits(self, estimation_result: EstimationResult, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot number of successful fits by distribution.
        
        Args:
            estimation_result: Results from estimation
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        
        # Get data for plotting
        distros = estimation_result.distros
        n_success = [estimation_result.series_results[distro].n_success for distro in distros]
        total_windows = len(estimation_result.series_results[distros[0]].fit_results) if distros else 0
        
        # Create bar plot
        x_pos = np.arange(len(distros))
        bars = ax.bar(x_pos, n_success, color=self.colors[:len(distros)], alpha=0.7)
        
        # Add labels for each bar
        for i, (bar, count) in enumerate(zip(bars, n_success)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(count), ha='center', va='bottom')
        
        # Add total line
        ax.axhline(total_windows, color="red", linestyle="--", 
                  label=f"Total Windows ({total_windows})", linewidth=2)
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels([d.capitalize() for d in distros], rotation=45)
        ax.set_ylabel("Number of Successful Fits")
        ax.set_title("Number of Successful Fits by Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def visualize_fit_deconvolution(self, estimation_result: EstimationResult,
                                  series_data, params: EstimationParams,
                                  real_los: Optional[np.ndarray] = None,
                                  df_mutant_selection: Optional[pd.DataFrame] = None,
                                  window_id: int = 2,
                                  save_path: Optional[str] = None,
                                  hide_failed: bool = True) -> plt.Figure:
        """
        Create comprehensive deconvolution visualization for a specific window.
        
        Args:
            estimation_result: Results from estimation
            series_data: Time series data
            params: Estimation parameters
            real_los: Real LOS distribution for comparison
            df_mutant_selection: Mutation/variant data
            window_id: Which window to visualize
            save_path: Path to save figure
            hide_failed: Whether to hide failed fits
            
        Returns:
            matplotlib Figure
        """
        # Setup visualization data structures
        distro_colors = {
            'gamma': '#1f77b4',
            'lognorm': '#ff7f0e', 
            'weibull': '#2ca02c',
            'expon': '#d62728',
            'norm': '#9467bd',
            'beta': '#8c564b'
        }
        
        replace_names = {
            'gamma': 'Gamma',
            'lognorm': 'Log-Normal',
            'weibull': 'Weibull',
            'expon': 'Exponential',
            'norm': 'Normal',
            'beta': 'Beta'
        }
        
        SHOW_MUTANTS = df_mutant_selection is not None
        
        # Ensure window_id is valid
        if window_id >= len(series_data.window_infos):
            window_id = min(2, len(series_data.window_infos) - 1)
        
        window_info = series_data.window_infos[window_id]
        
        # Create figure layout
        if SHOW_MUTANTS:
            fig = plt.figure(figsize=(17, 12), dpi=150)
            gs = gridspec.GridSpec(3, 4, height_ratios=[5, 1, 3])
            ax_main = fig.add_subplot(gs[0, :4])
            ax_inc = ax_main.twinx()
            ax_kernel = fig.add_subplot(gs[2, :2])
            ax_err_train = fig.add_subplot(gs[2, 2])
            ax_err_test = fig.add_subplot(gs[2, 3])
            ax_mutant = fig.add_subplot(gs[1, :4])
        else:
            fig = plt.figure(figsize=(17, 10), dpi=150)
            gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1])
            ax_main = fig.add_subplot(gs[0, :4])
            ax_inc = ax_main.twinx()
            ax_kernel = fig.add_subplot(gs[1, :2])
            ax_err_train = fig.add_subplot(gs[1, 2])
            ax_err_test = fig.add_subplot(gs[1, 3])
        
        # Plot main ICU occupancy
        line_bedload, = ax_main.plot(series_data.y_full, color="black", 
                                   label="ICU Bedload", linewidth=2)
        
        # Plot training/test windows
        w = window_info
        span_los_cutoff = ax_main.axvspan(w.train_start, w.train_los_cutoff, 
                                        color="magenta", alpha=0.1,
                                        label=f"Train Window (Convolution Edge)")
        span_train = ax_main.axvspan(w.train_los_cutoff, w.train_end, 
                                   color="red", alpha=0.2,
                                   label=f"Training Window")
        span_test = ax_main.axvspan(w.test_start, w.test_end, 
                                  color="blue", alpha=0.1,
                                  label=f"Test Window")
        ax_main.axvline(w.train_end, color="black", linestyle="-", linewidth=1)
        
        # Plot admissions/incidence
        label = "New ICU Admissions (Scaled)" if params.fit_admissions else "COVID Incidence (Scaled)"
        line_inc, = ax_inc.plot(series_data.x_full, linestyle="--", 
                              label=label, alpha=0.7)
        ax_inc.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        
        # Set y-axis limits for incidence
        ma = np.nanmax(series_data.x_full)
        if ma > 0:
            ax_inc.set_ylim(-ma/7.5, ma*4)
        
        # Plot distribution fits
        for distro_name in estimation_result.distros:
            if distro_name not in distro_colors:
                continue
                
            result_series = estimation_result.series_results[distro_name]
            if window_id >= len(result_series.fit_results):
                continue
                
            result_obj = result_series.fit_results[window_id]
            if not result_obj or (hide_failed and not result_obj.success):
                continue
            
            color = distro_colors[distro_name]
            name = replace_names.get(distro_name, distro_name.capitalize())
            
            # Note: The original code expects result_obj to have 'kernel' and 'curve' attributes
            # These would need to be computed from the fit results in the actual implementation
            # For now, we'll plot placeholder data
            
            # Plot kernel (LOS distribution)
            if hasattr(result_obj, 'kernel') and result_obj.kernel is not None:
                ax_kernel.plot(result_obj.kernel, label=name, color=color)
            
            # Plot predicted curve
            if hasattr(result_obj, 'curve') and result_obj.curve is not None:
                y = result_obj.curve[params.los_cutoff:]
                s = np.arange(len(y)) + params.los_cutoff + w.train_start
                ax_main.plot(s, y, label=f"{name}", color=color, linewidth=2)
        
        # Plot real LOS if available
        if real_los is not None:
            ax_kernel.plot(real_los, color="black", label="Sentinel LoS", linewidth=2)
        
        # Plot error bars
        for i, distro_name in enumerate(estimation_result.distros):
            if distro_name not in distro_colors:
                continue
                
            result_series = estimation_result.series_results[distro_name]
            color = distro_colors[distro_name]
            
            if window_id < len(result_series.train_relative_errors):
                train_err = result_series.train_relative_errors[window_id]
                test_err = result_series.test_relative_errors[window_id]
                
                if hide_failed and not result_series.fit_results[window_id].success:
                    ax_err_train.bar(i, 1e100, color="lightgrey", hatch="/", alpha=0.5)
                    ax_err_test.bar(i, 1e100, color="lightgrey", hatch="/", alpha=0.5)
                    continue
                
                ax_err_train.bar(i, train_err, color=color, alpha=0.7)
                ax_err_test.bar(i, test_err, color=color, alpha=0.7)
        
        # Plot mutations if available
        if SHOW_MUTANTS and df_mutant_selection is not None:
            for col in df_mutant_selection.columns:
                line, = ax_mutant.plot(df_mutant_selection[col].values, alpha=0.7)
                ax_mutant.fill_between(range(len(df_mutant_selection)), 
                                     df_mutant_selection[col].values, 0, alpha=0.3)
            
            ax_mutant.legend(df_mutant_selection.columns, loc="upper right")
            ax_mutant.set_title("Variants of Concern")
            ax_mutant.set_ylabel("Variant Share (%)")
            ax_mutant.set_xlim(0, len(series_data.y_full))
        
        # Format main plot
        ax_main.set_title("ICU Occupancy Deconvolution")
        ax_main.set_ylabel("Occupied Beds")
        ax_main.set_xlim(0, len(series_data.y_full))
        ax_main.legend(loc="upper left")
        
        # Format incidence plot
        ax_inc.set_ylabel(label)
        
        # Format kernel plot
        ax_kernel.set_ylabel("Discharge Probability")
        ax_kernel.set_xlabel("Days after admission")
        ax_kernel.set_title("Estimated LoS Kernels")
        ax_kernel.set_ylim(0, 0.1)
        ax_kernel.set_xlim(-2, 80)
        ax_kernel.legend()
        
        # Format error plots
        lim = 0.4
        ax_err_train.set_ylim(0, lim)
        ax_err_train.set_title("Relative Train Error")
        ax_err_train.set_xticks(range(len(estimation_result.distros)))
        ax_err_train.set_xticklabels([replace_names.get(d, d.capitalize()) 
                                    for d in estimation_result.distros], rotation=45)
        ax_err_train.set_ylabel("Relative Error")
        
        ax_err_test.set_ylim(0, lim)
        ax_err_test.set_title("Relative Test Error")
        ax_err_test.set_xticks(range(len(estimation_result.distros)))
        ax_err_test.set_xticklabels([replace_names.get(d, d.capitalize()) 
                                   for d in estimation_result.distros], rotation=45)
        ax_err_test.set_ylabel("Relative Error")
        
        plt.suptitle(f"Deconvolution Training Process\nWindow {window_id}", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_error_failure_rate(self, estimation_result: EstimationResult, 
                              col1: str, col2: str, ylim: Optional[Tuple[float, float]] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot error vs failure rate scatter plot.
        
        Args:
            estimation_result: Results from estimation
            col1: X-axis column name
            col2: Y-axis column name
            ylim: Y-axis limits
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        # Get summary statistics
        summary_stats = estimation_result.get_summary_stats()
        
        for i, distro in enumerate(estimation_result.distros):
            if distro in ["sentinel", "block"]:
                continue
                
            # Map generic column names to actual summary stats columns
            col1_mapped = self._map_column_name(col1)
            col2_mapped = self._map_column_name(col2)
            
            if col1_mapped in summary_stats.columns and col2_mapped in summary_stats.columns:
                row = summary_stats[summary_stats['distribution'] == distro]
                if not row.empty:
                    val1 = row[col1_mapped].iloc[0]
                    val2 = row[col2_mapped].iloc[0]
                    
                    color = self.colors[i % len(self.colors)]
                    ax.scatter(val1, val2, s=100, label=distro.capitalize(), color=color)
                    ax.annotate(distro.capitalize(), (val1, val2), fontsize=9, 
                              xytext=(5, 5), textcoords='offset points')
        
        # Labels and formatting
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f"Model Performance: {col1} vs. {col2}")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def _map_column_name(self, col_name: str) -> str:
        """Map generic column names to actual summary stats columns."""
        mapping = {
            "Failure Rate Train": "failure_rate_train",
            "Failure Rate Test": "failure_rate_test", 
            "Median Loss Train": "median_train_error",
            "Median Loss Test": "median_test_error",
            "Mean Loss Train": "mean_train_error",
            "Mean Loss Test": "mean_test_error",
            "Mean Loss Test (no outliers)": "mean_test_error"
        }
        return mapping.get(col_name, col_name.lower().replace(" ", "_"))
    
    def plot_error_boxplots(self, estimation_result: EstimationResult,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot boxplots of training and test errors by distribution.
        
        Args:
            estimation_result: Results from estimation
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
        
        # Prepare data for boxplots
        train_errors_by_distro = []
        test_errors_by_distro = []
        distro_labels = []
        
        for distro in estimation_result.distros:
            series_result = estimation_result.series_results[distro]
            
            # Get valid errors (exclude failed fits)
            train_errors = [err for err, fit_result in zip(series_result.train_relative_errors, 
                                                         series_result.fit_results) 
                          if fit_result and fit_result.success and not np.isnan(err) and not np.isinf(err)]
            test_errors = [err for err, fit_result in zip(series_result.test_relative_errors, 
                                                        series_result.fit_results) 
                         if fit_result and fit_result.success and not np.isnan(err) and not np.isinf(err)]
            
            train_errors_by_distro.append(train_errors)
            test_errors_by_distro.append(test_errors)
            distro_labels.append(f"{distro.capitalize()} n={len(train_errors)}")
        
        # Plot boxplots
        bp1 = ax1.boxplot(train_errors_by_distro, labels=distro_labels, patch_artist=True)
        bp2 = ax2.boxplot(test_errors_by_distro, labels=distro_labels, patch_artist=True)
        
        # Color the boxes
        for i, (box1, box2) in enumerate(zip(bp1['boxes'], bp2['boxes'])):
            color = self.colors[i % len(self.colors)]
            box1.set_facecolor(color)
            box1.set_alpha(0.7)
            box2.set_facecolor(color)
            box2.set_alpha(0.7)
        
        # Format plots
        ax1.set_title("Training Error Distribution")
        ax1.set_ylabel("Relative Train Error")
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title("Test Error Distribution")
        ax2.set_ylabel("Relative Test Error")
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_error_stripplot(self, estimation_result: EstimationResult,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot stripplot of training errors by distribution.
        
        Args:
            estimation_result: Results from estimation
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        
        # Prepare data
        train_errors_by_distro = []
        for distro in estimation_result.distros:
            series_result = estimation_result.series_results[distro]
            train_errors = [err for err, fit_result in zip(series_result.train_relative_errors, 
                                                         series_result.fit_results) 
                          if fit_result and fit_result.success and not np.isnan(err) and not np.isinf(err)]
            train_errors_by_distro.append(train_errors)
        
        # Create stripplot
        try:
            sns.stripplot(data=train_errors_by_distro, jitter=0.2, ax=ax)
        except:
            # Fallback if seaborn stripplot fails
            for i, errors in enumerate(train_errors_by_distro):
                x = np.random.normal(i, 0.1, len(errors))
                ax.scatter(x, errors, alpha=0.6, color=self.colors[i % len(self.colors)])
        
        ax.set_xticks(range(len(estimation_result.distros)))
        ax.set_xticklabels([d.capitalize() for d in estimation_result.distros], rotation=45)
        ax.set_title("Training Error Distribution")
        ax.set_ylabel("Relative Training Error")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_individual_distribution_analysis(self, estimation_result: EstimationResult,
                                             series_data, distro_name: str,
                                             sentinel_range: Optional[Tuple[int, int]] = None,
                                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot detailed analysis for a single distribution.
        
        Args:
            estimation_result: Results from estimation
            series_data: Series data object
            distro_name: Name of distribution to analyze
            sentinel_range: Tuple of (start, end) for sentinel period
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        if distro_name not in estimation_result.series_results:
            raise ValueError(f"Distribution {distro_name} not found in results")
        
        series_result = estimation_result.series_results[distro_name]
        
        fig, (ax, ax4, ax2) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=150)
        
        # Plot actual data
        ax.plot(series_data.y_full, color="black", label="Real", alpha=0.8, linestyle="--")
        
        # Plot fits for each window
        for window_info, fit_result in zip(series_data.window_infos, series_result.fit_results):
            if not fit_result or not fit_result.success:
                ax.axvspan(window_info.train_start, window_info.train_end, 
                          color="red", alpha=0.1)
                continue
            
            # Plot training curve
            if hasattr(fit_result, 'curve') and fit_result.curve is not None:
                params = estimation_result.params
                y_train = fit_result.curve[params.los_cutoff:params.train_width]
                x_train = np.arange(window_info.train_los_cutoff, window_info.train_end)
                ax.plot(x_train, y_train, color=self.colors[0], alpha=0.7)
                
                # Plot prediction curve
                y_pred = fit_result.curve[params.train_width:params.train_width+params.test_width]
                x_pred = np.arange(window_info.train_end, window_info.test_end)
                ax.plot(x_pred, y_pred, color=self.colors[1], alpha=0.7)
        
        # Add legend elements
        ax.plot([], [], color=self.colors[0], label=f"{distro_name.capitalize()} Train")
        ax.plot([], [], color=self.colors[1], label=f"{distro_name.capitalize()} Prediction")
        ax.axvspan(0, 0, color="red", alpha=0.1, label="Failed Training Windows")
        
        if sentinel_range:
            ax.axvspan(sentinel_range[0], sentinel_range[1], 
                      color="green", alpha=0.1, label="Sentinel Window")
        
        ax.legend(loc="upper left")
        ax.set_ylim(-100, 6000)
        ax.grid(True, alpha=0.3)
        
        # Plot transition rates
        windows = np.arange(len(series_result.fit_results))
        transition_rates = [fr.params[0] if fr and fr.success and fr.params else np.nan 
                          for fr in series_result.fit_results]
        
        valid_mask = ~np.isnan(transition_rates)
        ax2.bar(windows[valid_mask], np.array(transition_rates)[valid_mask], 
               width=0.8, label="Transition Probability", alpha=0.7)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.01, 0.1)
        ax2.set_title("Transition Probability")
        ax2.set_ylabel("Probability")
        
        # Plot errors
        ax4.plot(windows, series_result.train_relative_errors, label="Train Error", marker='o')
        ax4.plot(windows, series_result.test_relative_errors, label="Test Error", marker='s')
        
        # Mark failed fits
        for i, fit_result in enumerate(series_result.fit_results):
            if not fit_result or not fit_result.success:
                ax4.axvline(i, color="red", alpha=0.5)
        
        ax4.axvline(-np.inf, color="red", label="Failed Fit", alpha=0.5)
        ax4.legend(loc="upper right")
        ax4.set_ylim(-100, 1000)  # Reasonable error range
        ax4.set_title("Error")
        ax4.set_ylabel("Relative Error")
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f"{distro_name.capitalize()} Distribution Analysis", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_all_predictions_combined(self, estimation_result: EstimationResult,
                                    series_data, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot all predictions from all distributions combined.
        
        Args:
            estimation_result: Results from estimation
            series_data: Series data object
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, dpi=150)
        
        # Plot actual ICU data
        ax.plot(series_data.y_full, color="black", label="ICU Bedload", linewidth=2)
        
        # Plot all predictions
        for i, distro_name in enumerate(estimation_result.distros):
            series_result = estimation_result.series_results[distro_name]
            color = self.colors[i % len(self.colors)]
            
            for window_info, fit_result in zip(series_data.window_infos, series_result.fit_results):
                if not fit_result or not fit_result.success:
                    continue
                
                if hasattr(fit_result, 'curve') and fit_result.curve is not None:
                    params = estimation_result.params
                    y_pred = fit_result.curve[params.train_width:params.train_width+params.test_width]
                    x_pred = np.arange(window_info.train_end, window_info.test_end)
                    ax.plot(x_pred, y_pred, color=color, alpha=0.6, linewidth=1)
        
        ax.set_ylim(-100, 6000)
        ax.set_title("All Predictions")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot transition rates for all distributions
        for i, distro_name in enumerate(estimation_result.distros):
            series_result = estimation_result.series_results[distro_name]
            windows = np.arange(len(series_result.fit_results))
            transition_rates = [fr.params[0] if fr and fr.success and fr.params else np.nan 
                              for fr in series_result.fit_results]
            
            valid_mask = ~np.isnan(transition_rates)
            if np.any(valid_mask):
                ax2.plot(windows[valid_mask], np.array(transition_rates)[valid_mask], 
                        label=distro_name.capitalize(), color=self.colors[i % len(self.colors)],
                        marker='o', alpha=0.7)
        
        ax2.legend(ncol=2, loc="upper right")
        ax2.set_ylim(-0.01, 0.075)
        ax2.set_title("Transition Rates")
        ax2.set_ylabel("Transition Rate")
        ax2.set_xlabel("Window")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_all_errors_combined(self, estimation_result: EstimationResult,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot all errors from all distributions in a single graph.
        
        Args:
            estimation_result: Results from estimation
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
        
        for i, distro_name in enumerate(estimation_result.distros):
            series_result = estimation_result.series_results[distro_name]
            windows = np.arange(len(series_result.fit_results))
            color = self.colors[i % len(self.colors)]
            
            # Plot train and test errors
            ax.plot(windows, series_result.train_relative_errors, 
                   label=f"{distro_name.capitalize()} Train", 
                   color=color, linestyle='-', marker='o', alpha=0.7)
            ax.plot(windows, series_result.test_relative_errors, 
                   label=f"{distro_name.capitalize()} Test", 
                   color=color, linestyle='--', marker='s', alpha=0.7)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title("All Errors Combined")
        ax.set_xlabel("Window")
        ax.set_ylabel("Relative Error")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def plot_distribution_kernels(self, estimation_result: EstimationResult,
                                real_los: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Plot estimated kernels (LOS distributions) for each distribution.
        
        Args:
            estimation_result: Results from estimation
            real_los: Real LOS distribution for comparison
            save_path: Base path to save figures (will append distribution name)
            
        Returns:
            Dictionary mapping distribution names to figures
        """
        figures = {}
        
        for distro_name in estimation_result.distros:
            series_result = estimation_result.series_results[distro_name]
            
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            
            # Plot real kernel if available
            if real_los is not None:
                ax.plot(real_los, color='black', label="Real LOS", linewidth=2)
            
            # Plot all estimated kernels
            for fit_result in series_result.fit_results:
                if fit_result and fit_result.success and hasattr(fit_result, 'kernel'):
                    if fit_result.kernel is not None:
                        ax.plot(fit_result.kernel, alpha=0.3, color=self.colors[0])
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(f"{distro_name.capitalize()} Kernel Distribution")
            ax.set_xlabel("Days after admission")
            ax.set_ylabel("Discharge Probability")
            ax.set_ylim(-0.005, 0.3)
            ax.set_xlim(0, 100)
            
            plt.tight_layout()
            
            if save_path:
                distro_path = f"{save_path}_{distro_name}_kernels.png"
                plt.savefig(distro_path, bbox_inches='tight')
            
            figures[distro_name] = fig
        
        return figures
