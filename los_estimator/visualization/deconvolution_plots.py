"""Deconvolution plotting functionality."""

import logging
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from ..config import (
    ModelConfig,
    OutputFolderConfig,
    VisualizationConfig,
    VisualizationContext,
)
from ..core import SeriesData
from ..fitting import MultiSeriesFitResults
from .base import VisualizerBase

logger = logging.getLogger("los_estimator")


class DeconvolutionPlots(VisualizerBase):
    """Plotting functionality for deconvolution analysis."""

    def __init__(
        self,
        all_fit_results: MultiSeriesFitResults,
        series_data: SeriesData,
        model_config: ModelConfig,
        visualization_config: VisualizationConfig,
        visualization_context: VisualizationContext,
        output_config: OutputFolderConfig,
    ):
        super().__init__(visualization_config, output_config)

        self.vc: VisualizationContext = visualization_context
        self.all_fit_results: MultiSeriesFitResults = all_fit_results
        self.series_data: SeriesData = series_data
        self.model_config: ModelConfig = model_config

    def plot_successful_fits(self):
        """Plot number of failed fits and successful fits."""
        fit_results = self.all_fit_results

        fig = self._figure()

        for i, distro in enumerate(fit_results):
            plt.bar(
                i,
                fit_results[distro].n_success,
                color=self.colors[i],
                label=distro.capitalize(),
            )

        plt.xticks(np.arange(len(fit_results)), fit_results.keys(), rotation=45)
        self._set_title("Number of successful fits\n")
        plt.axhline(
            self.series_data.n_windows, color="red", linestyle="--", label="Total"
        )
        plt.xticks(rotation=45)

        self._show("successful_fits.png", fig)

    def _pairplot(self, col2, col1):
        """Create scatter plot comparing two metrics."""
        name = f"{col2}_vs_{col1}"

        fig = self._figure()

        for i, distro in enumerate(self.all_fit_results.distros):
            if distro in ["sentinel", "block"]:
                continue
            val1 = self.all_fit_results.summary[col1][distro]
            val2 = self.all_fit_results.summary[col2][distro]
            plt.scatter(val1, val2, s=100, label=distro, color=self.colors[i])
            plt.annotate(
                distro,
                (val1, val2),
                fontsize=9,
                xytext=(5, 5),
                textcoords="offset points",
            )

        # Labels and formatting
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(True)
        self._set_title(f"Model Performance: {name}")
        self._show(name or f"{col2}_vs_{col1}.png", fig)

    def plot_err_failure_rates(self):
        """Plot error vs failure rate comparisons."""
        self._pairplot("Median Loss Train", "Failure Rate")
        self._pairplot("Median Loss Test", "Failure Rate")
        self._pairplot("Mean Loss Test (no outliers)", "Failure Rate")

    def plot_error_comparison(self):
        """Plot error comparison across models."""
        sorted_summary = self.all_fit_results.summary.sort_values("Median Loss Test")
        sorted_summary = sorted_summary[
            [
                "Median Loss Test",
                "Failure Rate",
                "Median Loss Train",
                "Upper Quartile Train",
                "Lower Quartile Train",
            ]
        ]

        sorted_summary.plot(subplots=True, figsize=(10, 10))

        plt.legend()
        plt.title("Median Loss")
        xticks = list(sorted_summary.index)
        plt.xticks(np.arange(len(xticks)), xticks, rotation=45)
        self._set_title("Error Comparison of Models")
        self._show("error_comparison.png")

    def boxplot_errors(self, errors, title, ylabel, file):
        """Create boxplot of errors."""
        self._figure()
        plt.boxplot(errors)
        distro_and_n = [
            f"{distro.capitalize()} n={fr.n_success}"
            for distro, fr in self.all_fit_results.items()
        ]
        plt.xticks(np.arange(len(distro_and_n)) + 1, distro_and_n, rotation=45)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        self._show(file)

    def stripplot_errors(self, title, file):
        """Create stripplot of errors."""
        self._figure()
        sns.stripplot(data=self.all_fit_results.train_errors_by_distro, jitter=0.2)
        plt.xticks(
            np.arange(len(self.all_fit_results)),
            self.all_fit_results.distros,
            rotation=45,
        )
        self._set_title(title)
        self._show(file)

    def _ax_plot_prediction_error_window(
        self, ax, fr_series, distro, error_window_alpha=0.1
    ):
        """Plot prediction error window on given axis."""
        ax.plot(self.series_data.y_full, color="black", alpha=0.8, linestyle="--")
        for w, fit_result in zip(fr_series.window_infos, fr_series.fit_results):

            if not fit_result.success and error_window_alpha > 0:
                ax.axvspan(
                    w.train_start, w.train_end, color="red", alpha=error_window_alpha
                )
                continue

            x = np.arange(w.train_los_cutoff, w.train_end)
            y = fit_result.curve[
                self.model_config.los_cutoff : self.model_config.train_width
            ]
            ax.plot(x, y, color=self.colors[0])

            x = np.arange(w.train_end, w.test_end)
            y = fit_result.curve[
                self.model_config.train_width : self.model_config.train_width
                + self.model_config.test_width
            ]
            ax.plot(x, y, color=self.colors[1])

        legend_handles = [
            plt.Line2D([0], [0], color="black", linestyle="--", label="Real"),
            plt.Line2D(
                [0], [0], color=self.colors[0], label=f"{distro.capitalize()} Train"
            ),
            plt.Line2D(
                [0],
                [0],
                color=self.colors[1],
                label=f"{distro.capitalize()} Prediction",
            ),
        ]
        if error_window_alpha > 0:
            legend_handles += [
                Patch(color="red", alpha=0.1, label="Failed Training Windows")
            ]
        ax.legend(handles=legend_handles, loc="upper right")

        ax.set_ylim(-100, 6000)
        ax.set_xticks(self.vc.xtick_pos[::2])
        ax.set_xticklabels(self.vc.xtick_label[::2])
        ax.set_xlim(*self.vc.xlims)
        ax.grid(zorder=0)

    def _ax_plot_error_error_points(self, ax2, fr_series, distro):
        """Plot error points on given axis."""
        logger.warning(
            "This function is deprecated and will be removed in future versions. _ax_plot_error_error_points"
        )
        return
        x = self.series_data.windows
        ax2.plot(x, fr_series.train_relative_errors, label="Train Error")
        ax2.plot(x, fr_series.test_relative_errors, label="Test Error")

        for i, fit_result in enumerate(fr_series.fit_results):
            if not fit_result.success:
                ax2.axvline(x[i], color="red", alpha=0.5)

        legend_handles = [
            plt.Line2D(
                [0], [0], color=self.colors[0], label=f"{distro.capitalize()} Train"
            ),
            plt.Line2D(
                [0],
                [0],
                color=self.colors[1],
                label=f"{distro.capitalize()} Prediction",
            ),
            plt.Line2D(
                [0], [0], color="red", label="Failed Training Windows", alpha=0.5
            ),
        ]
        ax2.legend(handles=legend_handles, loc="upper right")

        ax2.set_ylim(-0.1, 0.5)
        ax2.set_title("Relative Errors")
        ax2.grid(zorder=0)

    def show_error_windows(self, distro: Optional[Union[str, List[str]]] = None):
        """Show error windows for specified distributions."""
        distros = self._get_distro_as_array(distro)

        for distro in distros:
            fr_series = self.all_fit_results[distro]
            _, (ax, ax2) = self._get_subplots(2, 1, sharex=True, figsize=(12, 6))
            self._ax_plot_prediction_error_window(ax, fr_series, distro)
            self._ax_plot_error_error_points(ax2, fr_series, distro)

            plt.suptitle(
                f"{distro.capitalize()} Distribution\n{self.model_config.run_name}"
            )
            plt.tight_layout()

            self._show(f"prediction_error_{distro}_fit.png")

    def show_all_error_windows_superimposed(self):
        """Show all error windows superimposed."""
        _, (ax, ax2) = self._get_subplots(2, 1, sharex=True, figsize=(12, 6))
        for distro in self.all_fit_results.distros:
            fr_series = self.all_fit_results[distro]

            self._ax_plot_prediction_error_window(
                ax, fr_series, distro, error_window_alpha=0.05
            )
            self._ax_plot_error_error_points(ax2, fr_series, distro)

        for line in ax2.get_children():
            if isinstance(line, plt.Line2D):
                if line.get_color() == "red":
                    line.remove()

        self._set_title("All Predictions and Error")
        plt.tight_layout()
        self._show("prediction_error_all_distros.png")

    def _get_distro_as_array(
        self, distro: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        """Convert distro parameter to array format."""
        if distro is None:
            distros = self.all_fit_results.distros
        elif isinstance(distro, str):
            distros = [distro]
        else:
            distros = distro
        return distros

    def show_all_predictions(self):
        """Show all predictions together."""
        _, ax = self._get_subplots(1, 1, sharex=True, figsize=(15, 7.5))
        for distro in self.all_fit_results.distros:
            fr_series = self.all_fit_results[distro]

            self._ax_plot_prediction_error_window(
                ax, fr_series, distro, error_window_alpha=0
            )

        self._set_title("All Predictions")
        plt.tight_layout()
        self._show("prediction_all_distros.png")

    def superimpose_kernels(self, distro: Optional[Union[str, List[str]]] = None):
        """Show superimposed kernels for distributions."""
        distros = self._get_distro_as_array(distro)

        for distro in distros:
            self._figure(figsize=(10, 5))
            # plot real kernel
            (r,) = plt.plot(self.vc.real_los, color="black", label="Real")

            fit_results = self.all_fit_results[distro]
            for fit_result in fit_results.fit_results:
                if not fit_result.success:
                    continue
                (l,) = plt.plot(
                    fit_result.kernel,
                    alpha=0.3,
                    color=self.colors[0],
                    label="All Estimated",
                )
            plt.legend(handles=[r, l])
            plt.ylim(-0.005, 0.3)
            self._set_title(f"{distro.capitalize()} Kernel")
            plt.tight_layout()
            plt.grid()
            self._show(f"all_kernels_{distro}.png")

    def generate_plots_for_run(self):
        """Generate all plots for a run."""

        self.plot_successful_fits()
        self.plot_err_failure_rates()
        self.plot_error_comparison()
        self.boxplot_errors(
            self.all_fit_results.train_errors_by_distro,
            "Train Error",
            "Relative Train Error",
            "train_error_boxplot.png",
        )
        self.boxplot_errors(
            self.all_fit_results.test_errors_by_distro,
            "Test Error",
            "Relative Test Error",
            "test_error_boxplot.png",
        )
        self.stripplot_errors("Train Error", "train_error_stripplot.png")
        self.show_error_windows()
        self.show_all_error_windows_superimposed()
        self.show_all_predictions()
        self.superimpose_kernels()

    def _set_title(self, title: str, *args, **kwargs):
        """Set the title of the current figure."""
        run_name = self.model_config.run_name
        plt.title(title + "\n" + run_name, *args, **kwargs)
