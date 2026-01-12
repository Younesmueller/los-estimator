# %%
"""Animation functionality for deconvolution analysis."""

import logging
import os
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from los_estimator.config import (
    AnimationConfig,
    ModelConfig,
    OutputFolderConfig,
    VisualizationConfig,
    VisualizationContext,
)
from los_estimator.core import SeriesData
from los_estimator.fitting import MultiSeriesFitResults
from los_estimator.visualization.deconvolution_plots import DeconvolutionPlots

logger = logging.getLogger("los_estimator")


class DeconvolutionAnimator(DeconvolutionPlots):
    """Animation functionality for deconvolution analysis."""

    def __init__(
        self,
        all_fit_results: MultiSeriesFitResults,
        series_data: SeriesData,
        model_config: ModelConfig,
        visualization_config: VisualizationConfig,
        visualization_context: VisualizationContext,
        output_folder_config: OutputFolderConfig,
        animation_config: AnimationConfig,
        window_ids: Optional[list[int]] = None,
    ):
        super().__init__(
            all_fit_results,
            series_data,
            model_config,
            visualization_config,
            visualization_context,
            output_config=output_folder_config,
        )
        self.window_ids = window_ids
        self.ac = animation_config
        self._generate_animation_context()

    def _generate_animation_context(self):
        """Generate context for animation frames."""
        ac = self.ac
        ac.distro_colors = {
            distro: self.visualization_config.colors[i] for i, distro in enumerate(self.all_fit_results)
        }
        d = dict(ac.alternative_names)
        ac.distro_patches = [
            Patch(color=ac.distro_colors[distro], label=d.get(distro, distro.capitalize()))
            for distro in self.all_fit_results
        ]
        d = dict(ac.replace_short_names)
        self.ac.short_distro_names = [d.get(distro, distro) for distro in self.all_fit_results]

    def _get_subplots(self, SHOW_MUTANTS):
        """Get subplot configuration for animation."""
        fig = self._figure(figsize=(17, 10))
        if SHOW_MUTANTS:
            gs = gridspec.GridSpec(3, 4, height_ratios=[5, 1, 3])
            ax_main = fig.add_subplot(gs[0, :4])
            ax_kernel = fig.add_subplot(gs[2, :2])
            ax_err_train = fig.add_subplot(gs[2, 2])
            ax_err_test = fig.add_subplot(gs[2, 3])
            ax_mutant = fig.add_subplot(gs[1, :4])
        else:
            gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1])
            ax_main = fig.add_subplot(gs[0, :4])
            ax_kernel = fig.add_subplot(gs[1, :2])
            ax_err_train = fig.add_subplot(gs[1, 2])
            ax_err_test = fig.add_subplot(gs[1, 3])
            ax_mutant = None

        return fig, ax_main, ax_kernel, ax_err_train, ax_err_test, ax_mutant

    def _create_animation_folder(self):
        """Create folder for animation frames."""
        path = self.output_config.animation
        if os.path.exists(path):
            import shutil

            shutil.rmtree(path)
        os.makedirs(path)

    def _plot_ax_main(self, ax_main, window_id):
        """Plot main axis for animation frame."""
        w, ac, y_full, x_full = (
            self.series_data.get_window_info(window_id),
            self.ac,
            self.series_data.y_full,
            self.series_data.x_full,
        )

        (line_bedload,) = ax_main.plot(y_full, color="black", label="Ground Truth: ICU Bedload")

        span_train = ax_main.axvspan(
            w.train_start,
            w.train_end,
            color="yellow",
            alpha=0.2,
            label=f"Train Window = {self.model_config.train_width} days",
        )
        span_test = ax_main.axvspan(
            w.test_start,
            w.test_end,
            color="magenta",
            alpha=0.2,
            label=f"Test Window = {self.model_config.test_width} days",
        )
        ax_main.axvline(
            w.train_end,
            color="magenta",
            linestyle="-",
            linewidth=1,
            label="Prediction Start",
        )
        line_pred_start_vertical_marker = ax_main.axvline(
            -100,
            color="magenta",
            marker="|",
            linestyle="None",
            markersize=10,
            markeredgewidth=1.5,
            label="Prediction Start",
        )

        for distro, result_series in self.all_fit_results.items():
            if window_id >= len(result_series.fit_results):
                continue
            result_obj = result_series.fit_results[window_id]
            if self.ac.debug_hide_failed and not result_obj.success:
                continue

            y_train = result_obj.train_prediction[self.model_config.kernel_width :]
            y_test = result_obj.test_prediction[self.model_config.kernel_width :]
            x_train = np.arange(len(y_train)) + w.training_prediction_start
            x_test = np.arange(len(y_test)) + w.test_start

            ax_main.plot(x_train, y_train, linestyle="--", color=ac.distro_colors[distro])
            ax_main.plot(x_test, y_test, label=f"{distro.capitalize()}", color=ac.distro_colors[distro])
        (line_inc,) = ax_main.plot(x_full * 4, linestyle="--", label="ICU Admissions (Scaled * 4)")
        ma = np.nanmax(x_full)

        legend1 = ax_main.legend(handles=ac.distro_patches, loc="upper left", fancybox=True, ncol=2)
        legend2 = ax_main.legend(
            handles=[line_bedload, line_inc, span_train, line_pred_start_vertical_marker, span_test],
            loc="upper right",
        )

        ax_main.add_artist(legend1)
        ax_main.add_artist(legend2)

        ax_main.set_title("Model Results at Day " + str(window_id))
        ax_main.set_xticks(self.vc.xtick_pos)
        ax_main.set_xticklabels(self.vc.xtick_label)
        ax_main.set_xlim(*self.vc.xlims)
        ax_main.set_ylim(-200, 6000)
        ax_main.set_ylabel("Occupied Beds")

    def save_n_show_animation_frame(self, fig: plt.Figure, num: int):
        """Save the current figure as an animation frame."""

        if fig is None:
            fig = plt.gcf()

        if self.ac.save_figures:
            filename = os.path.join(self.vc.animation_folder, f"fit_{num:04d}.png")
            fig.savefig(filename, bbox_inches="tight")

        if self.ac.show_figures:
            plt.show(fig)
        else:
            plt.close(fig)

    def animate_fit_deconvolution(self, df_mutant: Optional[pd.DataFrame] = None):
        """Create animation of fit deconvolution process."""
        SHOW_MUTANTS = df_mutant is not None

        self._create_animation_folder()

        to_enumerate = list(enumerate(self.series_data.window_infos))
        if self.window_ids is not None:
            to_enumerate = [to_enumerate[i] for i in self.window_ids]
        window_counter = 1

        if self.ac.debug_animation:
            to_enumerate = to_enumerate[: min(3, len(to_enumerate))]
        n_windows = len(to_enumerate)
        for window_id, window_info in to_enumerate:
            logger.info(f"Animating window {window_counter}/{n_windows}")
            window_counter += 1

            w = window_info
            fig, ax_main, ax_kernel, ax_err_train, ax_err_test, ax_mutant = self._get_subplots(SHOW_MUTANTS)

            self._plot_ax_main(ax_main, window_id)
            self._plot_ax_kernel(ax_kernel, window_id)
            self._plot_ax_errors(ax_err_train, ax_err_test, window_id)
            if SHOW_MUTANTS:
                self._plot_ax_mutants(ax_mutant, df_mutant)

            plt.suptitle(
                f"{self.model_config.run_name.replace('_', ' ')}\n\nDeconvolution Training Process",
                fontsize=16,
            )

            plt.tight_layout()

            self.save_n_show_animation_frame(fig, num=window_info.train_end)

    def _get_max_errors(self):
        max_train_error = 0
        max_test_error = 0
        for window_id in range(len(self.series_data.window_infos)):
            for distro, fit_result in self.all_fit_results.items():
                if window_id >= len(fit_result.fit_results):
                    continue
                train_err = fit_result.train_errors[window_id]
                test_err = fit_result.test_errors[window_id]
                if train_err > max_train_error:
                    max_train_error = train_err
                if test_err > max_test_error:
                    max_test_error = test_err
        if self.ac.train_error_lim != "auto":
            max_train_error = self.ac.train_error_lim
        if self.ac.test_error_lim != "auto":
            max_test_error = self.ac.test_error_lim
        return max_train_error, max_test_error

    def _plot_ax_mutants(self, ax_mutant, df_mutant):
        """Plot mutant data on axis."""
        y_full = self.series_data.y_full
        mutant_lines = []
        for col in df_mutant.columns:
            (line,) = ax_mutant.plot(df_mutant[col].values)
            mutant_lines.append(line)
            ax_mutant.fill_between(range(len(y_full)), df_mutant[col].values, 0, alpha=0.3)

        ax_mutant.legend(mutant_lines, df_mutant.columns, loc="upper right")
        ax_mutant.set_xticks([])
        ax_mutant.set_xticklabels([])
        ax_mutant.set_xticks(self.vc.xtick_pos)
        tmp_xtick = [label.split("\n")[1:] for label in self.vc.xtick_label]
        tmp_xtick = [label[0] if label else "" for label in tmp_xtick]
        ax_mutant.set_xticklabels(tmp_xtick)
        ax_mutant.set_xlim(*self.vc.xlims)
        ax_mutant.set_title("Variants of Concern")
        ax_mutant.set_ylabel("Variant Share (%)")

    def _plot_ax_errors(self, ax_err_train, ax_err_test, window_id):
        """Plot error bars on axes."""
        train_error_lim, test_error_lim = self._get_max_errors()
        ac = self.ac
        for i, (distro, fit_result) in enumerate(self.all_fit_results.items()):
            if window_id >= len(fit_result.fit_results):
                continue
            c = ac.distro_colors[distro]
            train_err = fit_result.train_errors[window_id]
            test_err = fit_result.test_errors[window_id]

            ax_err_train.bar(i, train_err, label="Train", color=c)
            ax_err_test.bar(i, test_err, label="Test", color=c)

        error_fun = self.model_config.error_fun.capitalize()
        ax_err_train.set_ylim(0, train_error_lim * 1.1)
        ax_err_train.set_title("Relative Train Error")
        ax_err_train.set_xticks(range(len(self.all_fit_results)))
        ax_err_train.set_xticklabels(ac.short_distro_names, rotation=75)
        ax_err_train.set_ylabel(error_fun)

        ax_err_test.set_ylim(0, test_error_lim * 1.1)
        ax_err_test.set_title("Relative Test Error")
        ax_err_test.set_xticks(range(len(self.all_fit_results)))
        ax_err_test.set_xticklabels(ac.short_distro_names, rotation=75)
        ax_err_test.set_ylabel(error_fun)

    def _plot_ax_kernel(self, ax_kernel, window_id):
        """Plot kernel on axis."""
        ac = self.ac
        for distro, result_series in self.all_fit_results.items():
            if window_id >= len(result_series.fit_results):
                continue
            result_obj = result_series.fit_results[window_id]
            name = dict(ac.alternative_names).get(distro, distro.capitalize())
            if self.ac.debug_hide_failed and not result_obj.success:
                continue

            ax_kernel.plot(result_obj.kernel, label=name, color=ac.distro_colors[distro])

        sample_kernel_handle = ax_kernel.plot(self.vc.real_los, color="black", label="Sentinel LoS Charité")

        ax_kernel.legend(handles=sample_kernel_handle + ac.distro_patches, loc="upper right", fancybox=True, ncol=2)
        ax_kernel.set_ylim(0, 0.1)
        ax_kernel.set_xlim(-2, 60)
        ax_kernel.set_ylabel("Discharge Probability")
        ax_kernel.set_xlabel("Days After Admission")
        ax_kernel.set_title("Estimated LoS Kernels")


animator = DeconvolutionAnimator(
    all_fit_results=estimator.all_fit_results,
    series_data=estimator.series_data,
    model_config=estimator.model_config,
    visualization_config=estimator.visualization_config,
    visualization_context=estimator.visualization_context,
    animation_config=estimator.animation_config,
    output_folder_config=estimator.output_config,
    window_ids=estimator.fitter.chosen_windows,
)
animator.window_ids = [1, 5, 10]
animator.ac.show_figures = True
animator.animate_fit_deconvolution()
