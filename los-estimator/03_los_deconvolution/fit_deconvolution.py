
#%%
# TODO: Ich habe jetzt so viel daran rumgebaut und geaendert. Sind die Ergebnisse noch die gleichen??
# Die Ausführung ist viel zu schnell. DAs kann gar nicht sein.
# Vllt baue ich mit der orignalen Methode einen Test, der die Daten Vergleicht.

# reload imports
%load_ext autoreload
%autoreload 2
import os
import sys
import numpy as np
import pandas as pd
import types

from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append("../02_fit_los_distributions/")
from compartmental_model import calc_its_comp
from los_fitter import distributions, calc_its_convolution, fit_SEIR
from fit_deconvolution_functions import *
from core import *
from high_vizzz import *
import high_vizzz
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Let's Go!")
#%%
graph_colors = get_color_palette()

#%%

los_file = "../01_create_los_profiles/berlin/output_los/los_berlin_all.csv"

init_params_file = "../02_fit_los_distributions/output_los/los_berlin_all/fit_results.csv"
mutants_file = "../data/VOC_VOI_Tabelle.xlsx"

start_day = "2020-01-01"
end_day = "2025-01-01"

import utils

sentinel_start_date = pd.Timestamp("2020-10-01")
sentinel_end_date = pd.Timestamp("2021-06-21")
sentinel_start_day = utils.date_to_day(sentinel_start_date,start_day)
sentinel_end_day = utils.date_to_day(sentinel_end_date,start_day)

 #%%

df_occupancy, real_los, df_init, df_mutant, xtick_pos, xtick_label, new_icu_date, new_icu_day = load_all_data(los_file, init_params_file, mutants_file, start_day, end_day)


#%%

vis_context = VisualizationContext()
vis_context.xtick_pos = xtick_pos
vis_context.xtick_label = xtick_label
vis_context.real_los = real_los
vis_context.graph_colors = high_vizzz.get_color_palette()

vc = vis_context

#%%
from high_vizzz import InputDataVisualizer

# visualizer = InputDataVisualizer(vc, None)
# visualizer.show_input_data(sentinel_start_date, sentinel_end_date, df_occupancy, new_icu_date)
# visualizer.plot_icu_data(df_occupancy)
# df_mutant.plot()
# plt.show()


#%%



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


params.ideas = types.SimpleNamespace()
params.ideas.los_change_penalty = ["..."]
params.ideas.fitting_err = ["mse","mae","rel_err","weighted_mse","inv_rel_err","capacity_err","..."]
params.ideas.presenting_err = ["..."]



#%%


run_name = generate_run_name(params)
params.run_name = run_name

print("###################################################################")
print("Run Name:")
print(run_name)
print("###################################################################")


results_folder, figures_folder, animation_folder = create_result_folders(params.run_name)
#%%

series_data = SeriesData(df_occupancy, params, new_icu_day)

#%%
if params.fit_admissions:
    xlims = (new_icu_day-30, 1300)
else:
    xlims = (70, 1300)

vis_context.xlims = xlims
vis_context.results_folder = results_folder
vis_context.figures_folder = figures_folder
vis_context.animation_folder = animation_folder
#%%

from fit_deconvolution_produce_fitting_test_results import produce_fitting_test_result
from los_fitter import fit_SEIR, fit_kernel_to_series, SingleFitResult


def _compare_all_fitresults(all_fit_results, compare_all_fit_results):
    print("Starting comparison of all fit results...")

    all_successful = True

    for distro, fit_result in all_fit_results.items():
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

compare_window_data, compare_all_fit_results = produce_fitting_test_result(df_init, params, series_data,distributions,MultiSeriesFitResults,
    DEBUG_WINDOWS = False,
    DEBUG_DISTROS = False,
    ONLY_LINEAR = False,
    LESS_WINDOWS = True)


#%%
from los_fitter2 import fit_SEIR, fit_kernel_to_series as fit_kernel_to_series2, SingleFitResult

from fit_deconvolution_alt_classes import MultiSeriesFitter


init_parameters = defaultdict(list)
for distro, row in df_init.iterrows():
    init_parameters[distro] = row['params']
EXCLUDE_DISTROS = {"beta", "invgauss", "gamma", "weibull", "lognorm", "sentinel", "block"}
distros = [d for d in distributions if d  not in EXCLUDE_DISTROS]
multi_fitter = MultiSeriesFitter(series_data, params, distros, init_parameters)
multi_fitter.DEBUG_MODE(
        ONE_WINDOW = False,
        LESS_WINDOWS = True,
        LESS_DISTROS = False,
        ONLY_LINEAR = False,
)

window_data, all_fit_results = multi_fitter.fit()


_compare_all_fitresults(all_fit_results, compare_all_fit_results)
    
#%%
# Produce Test REsults
init_parameters = defaultdict(list)
for distro, row in df_init.iterrows():
    init_parameters[distro] = row['params']
EXCLUDE_DISTROS = {"beta", "invgauss", "gamma", "weibull", "lognorm", "sentinel", "block"}
distros = [d for d in distributions if d  not in EXCLUDE_DISTROS]
multi_fitter = MultiSeriesFitter(series_data, params, distros, init_parameters)
multi_fitter.DEBUG_MODE(
        ONE_WINDOW = False,
        LESS_WINDOWS = True,
        LESS_DISTROS = False,
        ONLY_LINEAR = False,
)

window_data, all_fit_results = multi_fitter.fit()
import pickle
# save window_data, all_fit_rsults
# Save window_data and all_fit_results to files
with open(os.path.join(results_folder, "all_fit_results_short.pkl"), "wb") as f:
    pickle.dump(all_fit_results, f)

print("window_data and all_fit_results have been saved.")
#%%
init_parameters = defaultdict(list)
for distro, row in df_init.iterrows():
    init_parameters[distro] = row['params']
EXCLUDE_DISTROS = {"beta", "invgauss", "gamma", "weibull", "lognorm", "sentinel", "block"}
distros = [d for d in distributions if d  not in EXCLUDE_DISTROS]
multi_fitter = MultiSeriesFitter(series_data, params, distros, init_parameters)
multi_fitter.DEBUG_MODE(
        ONE_WINDOW = False,
        LESS_WINDOWS = False,
        LESS_DISTROS = False,
        ONLY_LINEAR = False,
)

window_data, all_fit_results = multi_fitter.fit()
# save window_data, all_fit_rsults
# Save window_data and all_fit_results to files

with open(os.path.join(results_folder, "all_fit_results_long.pkl"), "wb") as f:
    pickle.dump(all_fit_results, f)
#%%
from high_vizzz import DeconvolutionPlots, DeconvolutionAnimator


deconv_plot_visualizer = DeconvolutionPlots(all_fit_results,series_data,params,vc)
deconv_plot_visualizer.generate_plots_for_run()
dpv = deconv_plot_visualizer

animator = DeconvolutionAnimator.from_deconvolution_plots(dpv)
animator.DEBUG_MODE()
animator.animate_fit_deconvolution(
    df_mutant
)

#%%
# Run models on a pulse

run_pulse_model(params.run_name, vc.animation_folder, all_fit_results, series_data, debug=True)

#%%


# Just fit SEIR-Model
from los_fitter import fit_SEIR
import pickle


class CompartmentalFitter:
    def __init__(self, initial_guess,window_data,params):
        self.initial_guess = initial_guess
        self.window_data = window_data
        self.params = params

    def fit(self, x_train, y_train, x_test, y_test):
        return fit_SEIR(x_train, y_train, x_test, y_test, self.initial_guess)
    def fit_predict_windows(self):
        y_preds = []
        for _, _, train_data, test_data in tqdm(self.window_data):
            _,result_obj = fit_SEIR(*train_data,*test_data, self.initial_guess,
                            los_cutoff=self.params.los_cutoff)
            x_test, y_test = test_data
            y_pred_b = calc_its_comp(x_test, *result_obj.params,y_test[0])
            y_preds.append(y_pred_b)
        return y_preds
    
initial_guess_comp = [1/7,0.02,0]
fitter = CompartmentalFitter(initial_guess_comp, window_data, params)
y_preds = fitter.fit_predict_windows()


fig,ax = plt.subplots(1,1,figsize=(10,7),sharex=True)
ax.plot(series_data.y_full,label="Real",color="black")
    
for (_, w, _, _), y_pred in zip(window_data, y_preds):
    xs2 = np.arange(w.train_end,w.test_end)
    ax.plot(xs2,y_pred[params.train_width:],color=graph_colors[1])
ax.set_xlim(*vc.xlims)
plt.title("SEIR-Models")
plt.show()

fig,ax = plt.subplots(1,1,figsize=(10,7),sharex=True)
ax.plot(series_data.y_full,label="Real",color="black")
for window_id, window_info, train_data, test_data in tqdm(window_data):
    w = window_info
    x_train, y_train = train_data
    x_test, y_test = test_data

    result_dict,result_obj = fit_SEIR(*train_data,*test_data, initial_guess_comp,
                      los_cutoff=params.los_cutoff)
    y_pred_b = calc_its_comp(x_test, *result_obj.params,y_test[0])
    xs2 = np.arange(w.train_end,w.test_end)
    ax.plot(xs2,y_pred_b[params.train_width:],color=graph_colors[1])
ax.set_xlim(*vc.xlims)
plt.title("SEIR-Models")
plt.show()
#%%


#%%
message = f"Finished Run - {params.run_name}"
print(message)
os.system(f'msg * "{message}"')

# %%
