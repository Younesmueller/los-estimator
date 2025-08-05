import os

import shutil
import time
from collections import defaultdict

from los_estimator.core import *
from los_estimator.data import DataLoader
from los_estimator.visualization import DeconvolutionPlots, DeconvolutionAnimator, InputDataVisualizer, VisualizationContext, get_color_palette
from los_estimator.fitting import MultiSeriesFitter



from los_estimator.core import *
from los_estimator.data import DataLoader
from los_estimator.visualization import DeconvolutionPlots, DeconvolutionAnimator, InputDataVisualizer, VisualizationContext, get_color_palette
from los_estimator.fitting import MultiSeriesFitter


class LosEstimationRun:
    def __init__(self,data_config,output_config,model_config,debug_configuration):
        self.data = None
        self.data_config = data_config
        self.output_config = output_config
        self.debug_configuration = debug_configuration
        self.model_config = model_config
        self.data_loader = DataLoader(data_config)
        self.visualization_context = VisualizationContext()
        self.input_visualizer = InputDataVisualizer(self.visualization_context)

    

    def load_data(self):
        self.data = self.data_loader.load_all_data()
        vc = self.visualization_context
        vc.xtick_pos = self.data.xtick_pos
        vc.xtick_label = self.data.xtick_label
        vc.real_los = self.data.real_los
        vc.graph_colors = get_color_palette()
        self.visualization_context = vc
        
        self.data_loaded = True

    def visualize_input_data(self):
        self.input_visualizer.data = self.data
        self.input_visualizer.show_input_data()
        self.input_visualizer.plot_icu_data( )
        self.input_visualizer.plot_mutant_data()
        
    def create_result_folders(self):
        c = self.output_config
        
        c.run_name = self.run_name
        c.build()

        if os.path.exists(c.results):
            shutil.rmtree(c.results)

        os.makedirs(c.results)
        os.makedirs(c.figures)
        os.makedirs(c.animation)

    def visualize_results(self):
        xlims = (-30, 725)

        vc = self.visualization_context
        vc.xlims = xlims
        vc.results_folder = self.output_config.results
        vc.figures_folder = self.output_config.figures
        vc.animation_folder = self.output_config.animation

        self.deconv_plot_visualizer = DeconvolutionPlots(self.all_fit_results,self.series_data,self.model_config,self.visualization_context)
        self.deconv_plot_visualizer.generate_plots_for_run()
        dpv = self.deconv_plot_visualizer

        animator = DeconvolutionAnimator.from_deconvolution_plots(dpv)
        animator.DEBUG_MODE()
        animator.animate_fit_deconvolution(
            self.data.df_mutant
        )

    def create_run(self):
        model_config = self.model_config
        timestamp = time.strftime("%y%m%d_%H%M")
        run_name = f"{timestamp}_dev"

        run_name+=f"_step{model_config.step}_train{model_config.train_width}_test{model_config.test_width}"
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
        model_config.run_name = run_name
        self.run_name = run_name

    def run_analysis(self,vis = True):
        self.create_run()
        self.create_result_folders()

        self.load_data()
        if vis:
            self.visualize_input_data()
        self.fit()
        if vis:
            self.visualize_results()
    
    def fit(self):
        col = "new_icu_smooth" if self.model_config.smooth_data else "new_icu"
        series_data = self.data.df_occupancy[col].values, self.data.df_occupancy["icu"].values
        self.series_data = SeriesData(*series_data, self.model_config)
        
        init_parameters = defaultdict(list)
        for distro, row in self.data.df_init.iterrows():
            init_parameters[distro] = row['params']
        
        multi_fitter = MultiSeriesFitter(self.series_data, self.model_config, self.model_config.distributions, init_parameters)
        multi_fitter.DEBUG_MODE(**self.debug_configuration.__dict__)

        self.window_data, self.all_fit_results = multi_fitter.fit()
        return self.window_data, self.all_fit_results
    def deine_mutter(self):
        col = "new_icu_smooth" if self.model_config.smooth_data else "new_icu"
        series_data = self.data.df_occupancy[col].values, self.data.df_occupancy["icu"].values
        self.series_data = SeriesData(*series_data, self.model_config)
        return self.series_data