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
from los_estimator.evaluation import EvaluationResult
from los_estimator.fitting import MultiSeriesFitResults
from los_estimator.visualization.base import VisualizerBase
from los_estimator.visualization.deconvolution_plots import DeconvolutionPlots


evaluator = estimator.evaluator
eval_res = evaluator.result
xtick_pos = estimator.data.xtick_pos
xtick_label = estimator.data.xtick_label
estimator.series_data.windows

# %%
colors = estimator.visualization_config.colors
windows = estimator.series_data.windows
for metric in eval_res.metric_names:
    plt.figure(figsize=(12, 6))
    for i_distro, distro in enumerate(eval_res.distros):
        train_res, test_res = eval_res.by_distro_and_metric(distro, metric)
        plt.plot(windows, test_res, label=f"{distro} - Test - {metric}", color=colors[i_distro])
    plt.title(f"{metric.capitalize()} Test Metric for Rolling Models")
    plt.xlabel("Model Training Date")
    plt.ylabel(f"Error ({metric})")
    xlim = plt.gca().get_xlim()
    plt.gca().set_xticks(xtick_pos[1:-2:2])
    plt.gca().set_xticklabels(xtick_label[1:-2:2])
    plt.xlim(xlim)
    plt.legend()
    plt.savefig(os.path.join(estimator.output_config.metrics, f"rolling_{metric}_test.png"))
    plt.show()
    break
# %%


class MetricsPlots(VisualizerBase):

    def __init__(
        self,
        series_data: SeriesData,
        visualization_config: VisualizationConfig,
        visualization_context: VisualizationContext,
        output_config: OutputFolderConfig,
        evaluation_results: EvaluationResult,
    ):
        super().__init__(visualization_config, output_config)
        self.vc: VisualizationContext = visualization_context
        self.series_data: SeriesData = series_data
        self.evaluation_results: EvaluationResult = evaluation_results

        self.output_path = output_config.metrics

    def plot_metrics(self):
        eval_res = self.evaluation_results

        for metric in eval_res.metric_names:
            fig = self._figure()

            for i_distro, distro in enumerate(eval_res.distros):
                train_res, test_res = eval_res.by_distro_and_metric(distro, metric)
                plt.plot(
                    self.series_data.windows, test_res, label=f"{distro} - Test - {metric}", color=colors[i_distro]
                )
            plt.title(f"{metric.capitalize()} Test Metric for Rolling Models")
            plt.xlabel("Model Training Date")
            plt.ylabel(f"Error ({metric})")
            xlim = plt.gca().get_xlim()
            plt.gca().set_xticks(self.vc.xtick_pos[1:-2:2])
            plt.gca().set_xticklabels(self.vc.xtick_label[1:-2:2])
            plt.xlim(xlim)
            plt.legend()
            self._show(f"{metric}_test.png")


# %%
metrics_plots = MetricsPlots(
    series_data=estimator.series_data,
    visualization_config=estimator.visualization_config,
    visualization_context=estimator.visualization_context,
    output_config=estimator.output_config,
    evaluation_results=estimator.evaluator.result,
)
metrics_plots.plot_metrics()
print("ok.")

# %%
