import logging
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import dill

from los_estimator.config import *
from los_estimator.core import *
from los_estimator.data import DataLoader, DataPackage
from los_estimator.evaluation import FitResultEvaluator
from los_estimator.fitting import MultiSeriesFitter
from los_estimator.fitting.fit_results import MultiSeriesFitResults
from los_estimator.visualization import (
    DeconvolutionAnimator,
    DeconvolutionPlots,
    InputDataVisualizer,
    get_color_palette,
)

logger = logging.getLogger("los_estimator")


class LosEstimationRun:
    def __init__(
        self,
        data_config: DataConfig,
        output_config: OutputFolderConfig,
        model_config: ModelConfig,
        debug_config: DebugConfig,
        visualization_config: VisualizationConfig,
        animation_config: AnimationConfig,
        run_nickname: str = None,
    ):
        self.configurations = [
            data_config,
            output_config,
            model_config,
            debug_config,
            visualization_config,
            animation_config,
        ]
        self.run_nickname = run_nickname
        self.model_config: ModelConfig = model_config
        self.output_config: OutputFolderConfig = output_config
        self.data_config: DataConfig = data_config
        self.debug_config: DebugConfig = debug_config
        self.visualization_config: VisualizationConfig = visualization_config
        self.animation_config: AnimationConfig = animation_config

        self.visualization_context: VisualizationContext = VisualizationContext()
        self.data: DataPackage = None

        self.create_run()
        output_config.run_name = self.run_name
        output_config.build()

        if self.visualization_config.colors is None:
            self.visualization_config.colors = get_color_palette()

        self.data_loader: DataLoader = DataLoader(data_config)

        self.input_visualizer: InputDataVisualizer = InputDataVisualizer(
            self.visualization_config, self.visualization_context
        )

        self.fitter: MultiSeriesFitter = None
        self.window_data: list = None
        self.all_fit_results: MultiSeriesFitResults = None
        self.series_data: SeriesData = None

        self.evaluators: dict[str, FitResultEvaluator] = None
        self.data_loaded = False

    def load_data(self):
        self.data = self.data_loader.load_all_data()

        vc = self.visualization_context
        vc.xtick_pos = self.data.xtick_pos
        vc.xtick_label = self.data.xtick_label
        vc.real_los = self.data.real_los

        vc.xlims = self.visualization_config.xlims
        vc.results_folder = self.output_config.results
        vc.figures_folder = self.output_config.figures
        vc.animation_folder = self.output_config.animation

        self.data_loaded = True

    def visualize_input_data(self):
        self.input_visualizer.data = self.data
        self.input_visualizer.show_input_data()
        self.input_visualizer.plot_icu_data()
        self.input_visualizer.plot_mutant_data()

    def set_up(self):
        c = self.output_config

        c.run_name = self.run_name
        c.build()

        if Path(c.results).exists():
            shutil.rmtree(c.results)

        Path(c.results).mkdir(parents=True, exist_ok=True)
        Path(c.figures).mkdir(parents=True, exist_ok=True)
        Path(c.animation).mkdir(parents=True, exist_ok=True)
        Path(c.metrics).mkdir(parents=True, exist_ok=True)

        self.set_up_logger()

    def set_up_logger(self):
        path = Path(self.output_config.results) / "run.log"
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    def visualize_results(self):
        if not self.visualization_config.show_figures and not self.visualization_config.save_figures:
            logger.info("Visualization is disabled. Skipping visualization.")
            return
        self.deconv_plot_visualizer = DeconvolutionPlots(
            self.all_fit_results,
            self.series_data,
            self.model_config,
            self.visualization_config,
            self.visualization_context,
            self.output_config,
        )
        self.deconv_plot_visualizer.generate_plots_for_run()

    def animate_results(self):
        if not self.animation_config.show_figures and not self.animation_config.save_figures:
            logger.info("Animation is disabled. Skipping animation creation.")
            return
        animator = DeconvolutionAnimator(
            all_fit_results=self.all_fit_results,
            series_data=self.series_data,
            model_config=self.model_config,
            visualization_config=self.visualization_config,
            visualization_context=self.visualization_context,
            animation_config=self.animation_config,
            output_folder_config=self.output_config,
            window_ids=self.fitter.chosen_windows,
        )

        animator.animate_fit_deconvolution(self.data.df_mutant)

    def create_run(self):
        model_config = self.model_config
        timestamp = time.strftime("%y%m%d_%H%M")
        run_name = f"{timestamp}_dev"

        run_name += f"_step{model_config.step}_train{model_config.train_width}_test{model_config.test_width}"
        run_name += "_fit_admissions"
        if model_config.smooth_data:
            run_name += "_smoothed"
        else:
            run_name += "_unsmoothed"
        run_name += "_" + model_config.error_fun
        if model_config.reuse_last_parametrization:
            run_name += "_reuse_last_parametrization"
        if model_config.variable_kernels:
            run_name += "_variable_kernels"

        run_name += f"_{self.run_nickname}" if self.run_nickname else ""
        model_config.run_name = run_name
        self.run_name = run_name

    def run_analysis(self, vis=False):

        self.set_up()
        self.load_data()

        if vis:
            self.visualize_input_data()
        self.fit()

        self.evaluate()
        self.save_results()

        self.visualize_results()

        self.animate_results()

    def fit(self):
        col = "new_icu_smooth" if self.model_config.smooth_data else "new_icu"
        series_data = (
            self.data.df_occupancy[col].values,
            self.data.df_occupancy["icu"].values,
        )
        self.series_data = SeriesData(*series_data, self.model_config)

        init_parameters = defaultdict(list)
        for distro, row in self.data.df_init.iterrows():
            init_parameters[distro] = row["params"]

        self.fitter = MultiSeriesFitter(
            self.series_data,
            self.model_config,
            self.model_config.distributions,
            init_parameters,
        )
        self.fitter.DEBUG_MODE(self.debug_config)

        self.window_data, self.all_fit_results = self.fitter.fit()
        return self.window_data, self.all_fit_results

    def evaluate(self):
        if self.all_fit_results is None:
            raise ValueError("No fit results available. Please run the fit method first.")
        self.evaluators = {}
        for distro, fit_result in self.all_fit_results.items():
            evaluator = FitResultEvaluator(distro, self.series_data.y_full, fit_result.prediction)
            evaluator.evaluate()
            self.evaluators[distro] = evaluator
        for distro, evaluator in self.evaluators.items():
            logger.info(f"Evaluating {distro} distribution")

    def save_results(self):
        to_save = {
            "series_data": self.fitter.series_data,
            "chosen_windows": self.fitter.chosen_windows,
            "all_fit_results": self.all_fit_results,
        }

        for name, evaluator in self.evaluators.items():
            evaluator.save(self.output_config.metrics)

        for name, data in to_save.items():
            path = os.path.join(self.output_config.results, f"{name}.pkl")
            with open(path, "wb") as f:
                dill.dump(data, f)

        path = os.path.join(self.output_config.results, "configurations.toml")
        save_configurations(path, self.configurations)
