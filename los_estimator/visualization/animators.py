"""Animation functionality for deconvolution analysis."""

import os
import types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from typing import Optional

from .deconvolution_plots import DeconvolutionPlots
from ..core import SeriesData
from ..fitting import MultiSeriesFitResults
from ..config import ModelConfig, AnimationConfig, VisualizationConfig, VisualizationContext

from attr import dataclass

class DeconvolutionAnimator(DeconvolutionPlots):
    """Animation functionality for deconvolution analysis."""

    def __init__(self,
                all_fit_results: MultiSeriesFitResults,
                series_data: SeriesData, 
                model_config: ModelConfig,
                visualization_config: VisualizationConfig,
                visualization_context: VisualizationContext,
                animation_config: AnimationConfig):
        super().__init__(all_fit_results, series_data, model_config, visualization_config, visualization_context)
        self.ac = animation_config
        self._generate_animation_context()
    

    def _generate_animation_context(self):    
        """Generate context for animation frames."""
        ac = self.ac
        ac.distro_colors = {distro: self.visualization_config.colors[i] for i, distro in enumerate(self.all_fit_results)}
        ac.distro_patches = [
            Patch(color=ac.distro_colors[distro], label=ac.alternative_names.get(distro, distro.capitalize()))
            for distro in self.all_fit_results
        ]
        self.ac.short_distro_names = [ac.replace_short_names.get(distro,distro) for distro in self.all_fit_results]


    def _get_subplots(self, SHOW_MUTANTS):
        """Get subplot configuration for animation."""
        fig = self._figure(figsize=(17, 10))
        if SHOW_MUTANTS:
            gs = gridspec.GridSpec(3, 4, height_ratios=[5, 1, 3])
            ax_main = fig.add_subplot(gs[0, :4])
            ax_inc = ax_main.twinx()
            ax_kernel = fig.add_subplot(gs[2, :2])
            ax_err_train = fig.add_subplot(gs[2, 2])
            ax_err_test = fig.add_subplot(gs[2, 3])
            ax_mutant = fig.add_subplot(gs[1, :4])
        else:
            gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1])
            ax_main = fig.add_subplot(gs[0, :4])
            ax_inc = ax_main.twinx()
            ax_kernel = fig.add_subplot(gs[1, :2])
            ax_err_train = fig.add_subplot(gs[1, 2])
            ax_err_test = fig.add_subplot(gs[1, 3])
            ax_mutant = None
            
        return ax_main, ax_inc, ax_kernel, ax_err_train, ax_err_test, ax_mutant
    
    def _create_animation_folder(self):
        """Create folder for animation frames."""
        path = self.vc.output_folder_config.animation
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path)

    def _plot_ax_main(self, ax_main, ax_inc, window_id):
        """Plot main axis for animation frame."""
        w, ac, y_full, x_full = (self.series_data.get_window_info(window_id), 
                                self.ac, 
                                self.series_data.y_full, 
                                self.series_data.x_full)

        line_bedload, = ax_main.plot(y_full, color="black", label="ICU Bedload")

        span_los_cutoff = ax_main.axvspan(w.train_start, w.train_los_cutoff, color="magenta", alpha=0.1,
                                         label=f"Train Window (Convolution Edge) = {self.model_config.train_width} days")
        span_train = ax_main.axvspan(w.train_los_cutoff, w.train_end, color="red", alpha=0.2,
                                    label=f"Training = {self.model_config.train_width-self.model_config.los_cutoff} days")
        span_test = ax_main.axvspan(w.test_start, w.test_end, color="blue", alpha=0.05,
                                   label=f"Test Window = {self.model_config.test_width} days")
        ax_main.axvline(w.train_end, color="black", linestyle="-", linewidth=1)

        for distro, result_series in self.all_fit_results.items():
            if window_id >= len(result_series.fit_results):
                continue
            result_obj = result_series.fit_results[window_id]
            if self.ac.DEBUG_HIDE_FAILED and not result_obj.success:
                continue
            
            y = result_obj.curve[self.model_config.los_cutoff:]
            s = np.arange(len(y)) + self.model_config.los_cutoff + w.train_start
            ax_main.plot(s, y, label=f"{distro.capitalize()}", color=ac.distro_colors[distro])

        label = "New ICU Admissions (Scaled)"

        line_inc, = ax_inc.plot(x_full, linestyle="--", label=label)
        ax_inc.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ma = np.nanmax(x_full)
        ax_inc.set_ylim(-ma/7.5, ma*4)

        legend1 = ax_main.legend(handles=ac.distro_patches, loc="upper left", fancybox=True, ncol=2)
        legend2 = ax_main.legend(handles=[line_bedload, line_inc, span_los_cutoff, span_train, span_test], 
                                loc="upper right")

        ax_main.add_artist(legend1)
        ax_main.add_artist(legend2)

        ax_main.set_title("ICU Occupancy")
        ax_main.set_xticks(self.vc.xtick_pos)
        ax_main.set_xticklabels(self.vc.xtick_label)
        ax_main.set_xlim(*self.vc.xlims)
        ax_main.set_ylim(-200, 6000)
        ax_main.set_ylabel("Occupied Beds")

        ax_inc.set_ylabel("New ICU Admissions (scaled)")

    def save_n_show_animation_frame(self, fig: plt.Figure, window_id: int):
        """Save the current figure as an animation frame."""
        if self.DEBUG_ANIMATION:
            plt.show()
        else:
            fig.savefig(self.vc.animation_folder + f"fit_{window_id:04d}.png")
            plt.close(fig)
        plt.clf()

    def _show(self, filename: str = None, fig: Optional[plt.Figure] = None):
        """Save the figure and show it."""
        if fig is None:
            fig = plt.gcf()

        if self.ac.save_figures:
            if filename and not filename.endswith('.png'):
                filename = filename + '.png'
            fig.savefig(self.figures_folder + filename, bbox_inches='tight')

        if self.ac.show_figures:
            plt.show()
        else:
            plt.clf()

    def animate_fit_deconvolution(self, df_mutant: Optional[pd.DataFrame] = None):        
        """Create animation of fit deconvolution process."""
        SHOW_MUTANTS = df_mutant is not None

        if not self.ac.DEBUG_ANIMATION:
            self._create_animation_folder()

        window_counter = 1
        to_enumerate = list(enumerate(self.series_data.window_infos))

        if self.ac.DEBUG_ANIMATION:
            to_enumerate = [to_enumerate[min(2, len(to_enumerate)-1)]]

        for window_id, window_info in to_enumerate:
            print(f"Animation Window {window_counter}/{self.series_data.n_windows}")
            window_counter += 1

            w = window_info
            ax_main, ax_inc, ax_kernel, ax_err_train, ax_err_test, ax_mutant = self._get_subplots(SHOW_MUTANTS)
      
            self._plot_ax_main(ax_main, ax_inc, window_id)
            self._plot_ax_kernel(ax_kernel, window_id)
            self._plot_ax_errors(ax_err_train, ax_err_test, window_id)
            if SHOW_MUTANTS:
                self._plot_ax_mutants(ax_mutant, df_mutant)

            plt.suptitle(f"Deconvolution Training Process\n{self.model_config.run_name.replace('_', ' ')}", fontsize=16)

            plt.tight_layout()
            if self.ac.DEBUG_ANIMATION:
                plt.show()
            else:
                plt.savefig(self.vc.animation_folder / f"fit_{w.train_start:04d}.png")
                plt.close()
            plt.clf()

    def _plot_ax_mutants(self, ax_mutant, df_mutant):
        """Plot mutant data on axis."""
        y_full = self.series_data.y_full
        mutant_lines = []
        for col in df_mutant.columns:
            line, = ax_mutant.plot(df_mutant[col].values)
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
        ac = self.ac
        for i, (distro, fit_result) in enumerate(self.all_fit_results.items()):
            if window_id >= len(fit_result.fit_results):
                continue
            c = ac.distro_colors[distro]
            train_err = fit_result.train_relative_errors[window_id]
            test_err = fit_result.test_relative_errors[window_id]

            if self.ac.DEBUG_HIDE_FAILED and not fit_result[window_id].success:
                ax_err_train.bar(i, 1e100, color="lightgrey", hatch="/")
                ax_err_test.bar(i, 1e100, color="lightgrey", hatch="/")
                ax_err_train.bar(i, train_err, color="black")
                ax_err_test.bar(i, test_err, color="black")
                continue
            ax_err_train.bar(i, train_err, label="Train", color=c)
            ax_err_test.bar(i, test_err, label="Test", color=c)

        lim = .4
        ax_err_train.set_ylim(0, lim)
        ax_err_train.set_title("Relative Train Error")
        ax_err_train.set_xticks(range(len(self.all_fit_results)))
        ax_err_train.set_xticklabels(ac.short_distro_names, rotation=75)
        ax_err_train.set_ylabel("Relative Error")

        ax_err_test.set_ylim(0, lim)
        ax_err_test.set_title("Relative Test Error")
        ax_err_test.set_xticks(range(len(self.all_fit_results)))
        ax_err_test.set_xticklabels(ac.short_distro_names, rotation=75)
        ax_err_test.set_ylabel("Relative Error")

    def _plot_ax_kernel(self, ax_kernel, window_id):
        """Plot kernel on axis."""
        ac = self.ac
        for distro, result_series in self.all_fit_results.items():
            if window_id >= len(result_series.fit_results):
                continue
            result_obj = result_series.fit_results[window_id]
            name = ac.alternative_names.get(distro, distro.capitalize())
            if self.ac.DEBUG_HIDE_FAILED and not result_obj.success:
                continue
                
            ax_kernel.plot(result_obj.kernel, label=name, color=ac.distro_colors[distro])

        ax_kernel.plot(self.vc.real_los, color="black", label="Sentinel LoS Charité")
            
        ax_kernel.legend(handles=ac.distro_patches, loc="upper right", fancybox=True, ncol=2)
        ax_kernel.set_ylim(0, 0.1)
        ax_kernel.set_xlim(-2, 80)
        ax_kernel.set_ylabel("Discharge Probability")
        ax_kernel.set_xlabel("Days after admission")
        ax_kernel.set_title("Estimated LoS Kernels")
