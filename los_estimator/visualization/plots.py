"""
Visualization utilities for LOS estimation results.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path


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
        plt.rcParams["savefig.facecolor"] = "white"
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["figure.dpi"] = 100

    def _get_color_palette(self) -> List[str]:
        """Get extended color palette."""
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors += [
            "#FFA07A",
            "#20B2AA",
            "#FF6347",
            "#808000",
            "#FF00FF",
            "#FFD700",
            "#00FF00",
            "#00FFFF",
            "#0000FF",
            "#8A2BE2",
        ]
        return colors

    def plot_time_series_overview(
        self, df_occupancy: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot overview of time series data (incidences and ICU occupancy).

        Args:
            df_occupancy: DataFrame with occupancy data
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        # Plot ICU occupancy
        df_occupancy["icu"].plot(
            ax=axes[0], label="ICU Occupancy", color=self.colors[0]
        )
        axes[0].set_title("ICU Bed Occupancy")
        axes[0].set_ylabel("Number of Beds")
        axes[0].legend()

        # Plot new ICU admissions
        df_occupancy["new_icu"].plot(
            ax=axes[1], label="New ICU (raw)", color=self.colors[1], alpha=0.7
        )
        if "new_icu_smooth" in df_occupancy.columns:
            df_occupancy["new_icu_smooth"].plot(
                ax=axes[1],
                label="New ICU (7-day avg)",
                color=self.colors[2],
                linewidth=2,
            )
        axes[1].set_title("Daily New ICU Admissions")
        axes[1].set_ylabel("Number of Admissions")
        axes[1].legend()

        # Format x-axis
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_distribution_comparison(
        self,
        estimation_result,
        metric: str = "test_error",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
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

        # This is a placeholder - would need actual EstimationResult implementation
        # to populate with real data

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_fit_quality_heatmap(
        self, estimation_result, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of fit quality across distributions and windows.

        Args:
            estimation_result: Results from estimation
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # This is a placeholder - would need actual EstimationResult implementation
        # to populate with real data

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_successful_fits(
        self, estimation_result, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot number of successful fits by distribution.

        Args:
            estimation_result: Results from estimation
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

        # This is a placeholder - would need actual data to plot
        ax.set_xlabel("Distribution")
        ax.set_ylabel("Number of Successful Fits")
        ax.set_title("Number of Successful Fits by Distribution")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_error_boxplots(
        self, estimation_result, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot boxplots of training and test errors by distribution.

        Args:
            estimation_result: Results from estimation
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=150)

        # This is a placeholder - would need actual data to plot
        ax1.set_title("Training Error Distribution")
        ax1.set_ylabel("Relative Train Error")

        ax2.set_title("Test Error Distribution")
        ax2.set_ylabel("Relative Test Error")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_error_stripplot(
        self, estimation_result, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot stripplot of training errors by distribution.

        Args:
            estimation_result: Results from estimation
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

        # This is a placeholder - would need actual data to plot
        ax.set_title("Training Error Distribution")
        ax.set_ylabel("Relative Training Error")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    def visualize_fit_deconvolution(
        self,
        estimation_result,
        series_data,
        model_config,
        real_los: Optional[np.ndarray] = None,
        df_mutant_selection: Optional[pd.DataFrame] = None,
        window_id: int = 2,
        save_path: Optional[str] = None,
        hide_failed: bool = True,
    ) -> plt.Figure:
        """
        Create comprehensive deconvolution visualization for a specific window.

        Args:
            estimation_result: Results from estimation
            series_data: Time series data
            model_config: Estimation parameters
            real_los: Real LOS distribution for comparison
            df_mutant_selection: Mutation/variant data
            window_id: Which window to visualize
            save_path: Path to save figure
            hide_failed: Whether to hide failed fits

        Returns:
            matplotlib Figure
        """
        SHOW_MUTANTS = df_mutant_selection is not None

        # Create figure layout
        if SHOW_MUTANTS:
            fig = plt.figure(figsize=(17, 12), dpi=150)
            gs = gridspec.GridSpec(3, 4, height_ratios=[5, 1, 3])
        else:
            fig = plt.figure(figsize=(17, 10), dpi=150)
            gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1])

        # This is a placeholder - would need actual implementation
        # with series_data and estimation_result structures

        plt.suptitle(f"Deconvolution Training Process\nWindow {window_id}", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    def create_summary_report(
        self,
        estimation_result,
        output_dir: str,
        series_data=None,
        real_los: Optional[np.ndarray] = None,
        df_mutant_selection: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
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
        f = self.plot_distribution_comparison(
            estimation_result, save_path=str(comp_path)
        )
        saved_plots["distribution_comparison"] = str(comp_path)

        # Fit quality heatmap
        heatmap_path = output_path / "fit_quality_heatmap.png"
        f = self.plot_fit_quality_heatmap(
            estimation_result, save_path=str(heatmap_path)
        )
        saved_plots["fit_quality_heatmap"] = str(heatmap_path)

        return saved_plots
