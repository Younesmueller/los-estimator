#%% 
%load_ext autoreload
%autoreload 2
import os

import numpy as np
import pandas as pd
import types
from pathlib import Path
import shutil
import time
from collections import defaultdict
from los_estimator.core import *

from los_estimator.data import DataLoader
from los_estimator.visualization import DeconvolutionPlots, DeconvolutionAnimator, InputDataVisualizer, VisualizationContext, get_color_palette
from los_estimator.fitting import MultiSeriesFitter

from comparison_data_loader import load_comparison_data

from tqdm import tqdm
import matplotlib.pyplot as plt


print("Let's Go!")
#%%
less_windows = True
compare_all_fit_results = load_comparison_data(less_windows)
print("Comparison data loaded successfully.")
#%%
def _compare_all_fitresults(all_fit_results, compare_all_fit_results):
    print("Starting comparison of all fit results...")

    all_successful = True
    for distro in compare_all_fit_results.keys():
        if distro not in all_fit_results:
            print(f"❌ Distribution {distro} not found in comparison results.")
            all_successful = False
    for distro, fit_result in all_fit_results.items():
        if distro=="compartmental":
            continue
        if distro not in compare_all_fit_results:
            print(f"❌ Distribution {distro} not found in comparison results.")
            all_successful = False
            continue

        comp_fit_result = compare_all_fit_results[distro]

        train_error_diff = np.abs(fit_result.train_relative_errors.mean() - comp_fit_result.train_relative_errors.mean())
        test_error_diff = np.abs(fit_result.test_relative_errors.mean() - comp_fit_result.test_relative_errors.mean())

        if train_error_diff > 1e-4 or test_error_diff > 1e-4:
            print(f"❌ Comparison failed for distribution: {distro}")
            print(f"Train Error Difference: {train_error_diff:.4f}")
            print(f"Test Error Difference: {test_error_diff:.4f}")
            print("-" * 50)
            all_successful = False
        else:
            print(f"✅ Comparison successful for distribution: {distro}")

    if all_successful:
        print("✅ All distributions compared successfully!")
    else:
        print("❌ Some distributions failed the comparison.")

#%%

class LOSEstimator:
    def __init__(self,data_config,output_config,params,debug_configuration):
        self.data = None
        self.data_config = data_config
        self.output_config = output_config
        self.debug_configuration = debug_configuration
        self.params = params
        self.data_loader = DataLoader(data_config)
        self.visualization_context = VisualizationContext()
        self.input_visualizer = InputDataVisualizer(self.visualization_context)
        self.folder_context = None

    

    def load_data(self):
        dir = os.getcwd()
        os.chdir("C:\data\src\los-estimator\los-estimator\data")
        self.data = self.data_loader.load_all_data()
        os.chdir(dir)
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
        run_name = self.run_name
        self.folder_context = types.SimpleNamespace()
        f = self.folder_context
        f.base = Path(self.output_config.results_folder_base)
        f.results = f.base / run_name
        f.figures = f.results / "figures"
        f.animation = f.results / "animation"

        if os.path.exists(f.results):
            shutil.rmtree(f.results)

        os.makedirs(f.results)
        os.makedirs(f.figures)
        os.makedirs(f.animation)

    def visualize_results(self):
        xlims = (self.data.new_icu_day-30, 1300)

        vc = self.visualization_context
        vc.xlims = xlims
        vc.results_folder = self.folder_context.results
        vc.figures_folder = self.folder_context.figures
        vc.animation_folder = self.folder_context.animation

        self.deconv_plot_visualizer = DeconvolutionPlots(self.all_fit_results,self.series_data,self.params,self.visualization_context)
        self.deconv_plot_visualizer.generate_plots_for_run()
        dpv  = self.deconv_plot_visualizer

        animator = DeconvolutionAnimator.from_deconvolution_plots(dpv)
        animator.DEBUG_MODE()
        animator.animate_fit_deconvolution(
            self.data.df_mutant
        )

    def create_run(self):
        params = self.params
        timestamp = time.strftime("%y%m%d_%H%M")
        run_name = f"{timestamp}_dev"

        run_name+=f"_step{params.step}_train{params.train_width}_test{params.test_width}"
        if params.fit_admissions:
            run_name += "_fit_admissions"
        else:
            run_name += "_fit_incidence"
        if params.smooth_data:
            run_name += "_smoothed"
        else:
            run_name += "_unsmoothed"
        run_name += "_" + params.error_fun
        if params.reuse_last_parametrization:
            run_name += "_reuse_last_parametrization"
        if params.variable_kernels:
            run_name += "_variable_kernels"
        params.run_name = run_name
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
        self.series_data = SeriesData(self.data.df_occupancy, params, self.data.new_icu_day)

        init_parameters = defaultdict(list)
        for distro, row in self.data.df_init.iterrows():
            init_parameters[distro] = row['params']
        
        multi_fitter = MultiSeriesFitter(self.series_data, params, self.params.distributions, init_parameters)
        multi_fitter.DEBUG_MODE(**self.debug_configuration.__dict__)

        self.window_data, self.all_fit_results = multi_fitter.fit()
        return self.window_data, self.all_fit_results

data_config = types.SimpleNamespace()
data_config.los_file = "../01_create_los_profiles/berlin/output_los/los_berlin_all.csv"
data_config.init_params_file = "../02_fit_los_distributions/output_los/los_berlin_all/fit_results.csv"
data_config.mutants_file = "../data/VOC_VOI_Tabelle.xlsx"

data_config.start_day = "2020-01-01"
data_config.end_day = "2025-01-01"
data_config.sentinel_start_date = pd.Timestamp("2020-10-01")
data_config.sentinel_end_date = pd.Timestamp("2021-06-21")

output_config = types.SimpleNamespace()
output_config.results_folder_base = "./results"

params = Params()
params.kernel_width = 120
params.los_cutoff = 60 # Ca. 90% of all patients are discharged after 41 days
params.smooth_data = False
params.train_width = 42 + params.los_cutoff
params.test_width = 21 #28 *4
params.step = 7
params.fit_admissions = True
params.error_fun = "mse"# "weighted_mse"
params.reuse_last_parametrization = True
params.variable_kernels = True
params.distributions = [
    # "lognorm",
    # "weibull",
    "gaussian",
    "exponential",
    # "gamma",
    # "beta",
    "cauchy",
    "t",
    # "invgauss",
    "linear",
    # "block",
    # "sentinel",
    "compartmental",
]


params.ideas = types.SimpleNamespace()
params.ideas.los_change_penalty = ["..."]
params.ideas.fitting_err = ["mse","mae","rel_err","weighted_mse","inv_rel_err","capacity_err","..."]
params.ideas.presenting_err = ["..."]

debug_configuration = types.SimpleNamespace()
debug_configuration.ONE_WINDOW = False
debug_configuration.LESS_WINDOWS = less_windows
debug_configuration.LESS_DISTROS = False
debug_configuration.ONLY_LINEAR = False

estimator = LOSEstimator(data_config,output_config,params,debug_configuration)
estimator.run_analysis(vis=False)

_compare_all_fitresults(estimator.all_fit_results, compare_all_fit_results)
# %%
