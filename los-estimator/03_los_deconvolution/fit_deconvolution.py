# X Eliminate distro_to_fit
# X Eliminate all_fit_results_by_window
# eliminate all data structures that are not needed
# eliminate result_object
# Put visualisations into functions


#%%
# reload imports
%load_ext autoreload
%autoreload 2
import os
import sys
from typing import OrderedDict
import numpy as np
import pandas as pd
import types
import seaborn as sns
import shutil
import timeit

import functools
from matplotlib.patches import Patch
import time
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.append("../02_fit_los_distributions/")
from fit_deconvolution_visualizations import *
from dataprep import load_los, load_incidences, load_icu_occupancy, load_mutant_distribution
from compartmental_model import calc_its_comp
from los_fitter import distributions, calc_its_convolution, fit_SEIR
plt.rcParams['savefig.facecolor']='white'
from fit_deconvolution_functions import *
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Let's Go!")
#%%
print_figs = True
def show_plt(*args,**kwargs):
    # if in vscode interactive
    if print_figs:
        plt.show()
    else:
        plt.clf()
graph_colors = get_graph_colors()

#%%
los_file = "../01_create_los_profiles/berlin/output_los/los_berlin_all.csv"
init_params_file = "../02_fit_los_distributions/output_los/los_berlin_all/fit_results.csv"
mutants_file = "../data/VOC_VOI_Tabelle.xlsx"

start_day = "2020-01-01"
end_day = "2025-01-01"

def date_to_day(date):
    return (date - pd.Timestamp(start_day)).days
def day_to_date(day):
    return pd.Timestamp(start_day) + pd.Timedelta(days=day)

sentinel_start_date =pd.Timestamp("2020-10-01")
sentinel_end_date = pd.Timestamp("2021-06-21")
sentinel_start_day = date_to_day(sentinel_start_date)
sentinel_end_day = date_to_day(sentinel_end_date)

 #%%

df_occupancy, real_los, df_init, df_mutant, xtick_pos, xtick_label, new_icu_date = load_all_data(los_file, init_params_file, mutants_file, start_day, end_day)

df_mutant_selection = df_mutant.copy()
df_mutant_selection["Omikron_BA.1/2"] = df_mutant_selection["Omikron_BA.1"] + df_mutant_selection["Omikron_BA.2"]
df_mutant_selection["Omikron_BA.4/5"] = df_mutant_selection["Omikron_BA.4"] + df_mutant_selection["Omikron_BA.5"]
df_mutant_selection = df_mutant_selection[['Delta_AY.1','Omikron_BA.1/2','Omikron_BA.4/5']]

new_icu_day = date_to_day(new_icu_date)
#%%

visualization_data = types.SimpleNamespace()
visualization_data.xtick_pos = xtick_pos
visualization_data.xtick_label = xtick_label
visualization_data.real_los = real_los
visualization_data.show_plt = show_plt
visualization_data.graph_colors = graph_colors
vd = visualization_data

#%%
manual_transition_rates = get_manual_transition_rates(df_occupancy)

plt.plot(manual_transition_rates,label="Manual Transition Rates")
plt.title("Manual Transition Rates")
plt.xticks(vd.xtick_pos[::4],vd.xtick_label[::4])
show_plt()


#%%

axs = df_occupancy.plot(subplots=True)
for ax in axs:
    ax.axvspan(sentinel_start_date,sentinel_end_date, color="green", alpha=0.1,label="Sentinel")
axs[-1].axvline(new_icu_date,color="black",linestyle="--",label="First ICU")

plt.suptitle("Incidences and ICU Occupancy")
vd.show_plt()

#%%
fig,ax = plt.subplots(2,1,figsize=(10,5),sharex=True)

df_occupancy["new_icu_smooth"].plot(ax=ax[1],label="new_icu",color="orange")
df_occupancy["icu"].plot(ax=ax[0],label="AnzahlFall")
ax[0].set_title("Tägliche Neuzugänge ICU, geglättet")
ax[1].set_title("ICU Bettenbelegung")
plt.tight_layout()

plt.plot()

#%%
df_mutant.plot()
#%%

class Params (types.SimpleNamespace):
    pass

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

if params.fit_admissions:
    xlims = (new_icu_day-30, 1300)
else:
    xlims = (70, 1300)
if params.fit_admissions:
    start = new_icu_day + params.train_width
else:
    start = 0
windows = np.arange(start,len(df_occupancy)-params.kernel_width, params.step)
#%%


#%%
class WindowInfo:
    def __init__(self,window,params):
        self.window = window        
        self.train_end = self.window
        self.train_start = self.window - params.train_width
        self.train_los_cutoff = self.train_start + params.los_cutoff
        self.test_start = self.train_end
        self.test_end = self.test_start + params.test_width
        

        self.train_window = slice(self.train_start,self.train_end)
        self.train_test_window = slice(self.train_start,self.test_end)
        self.test_window = slice(self.test_start,self.test_end)

        self.params = params
    def __repr__(self):
        return f"WindowInfo(window={self.window}, train_start={self.train_start}, train_end={self.train_end}, test_start={self.test_start}, test_end={self.test_end})"

#%%
# Next steps:
# 1. Methode, die trainiert, basierend auf den Windows und einer distro...
# 2. Aufbrechen in Äußere Schleife- distros
# 3. Ist die Struktur, wie ich das in Variablen speichere sinnvoll?
#      - Nein, nimm distro als index
# 4. Umstellen der plots auf plotly

def select_series(df, params):
    if params.fit_admissions:
        col = "new_icu_smooth" if params.smooth_data else "new_icu"
    else:
        col = "AnzahlFall" if params.smooth_data else "daily"
    return df[col].values, df["icu"].values




#%%

class SeriesData:
    def __init__(self,df_occupancy,params):
        self.x_full,self.y_full = select_series(df_occupancy, params)
        self._calc_windows(params)
        self.n_days = len(self.x_full)

    def _calc_windows(self,params):
        start = 0
        if params.fit_admissions:
            start = new_icu_day + params.train_width
        self.windows = np.arange(start,len(self.x_full)-params.kernel_width, params.step)
        self.window_infos = [WindowInfo(window,params) for window in self.windows]
        self.n_windows = len(self.windows)

    @ functools.lru_cache
    def get_train_data(self, window_id:int):
        if window_id > len(self.windows):
            raise ValueError(f"Window ID {window_id} out of range for {len(self.windows)} windows.")
        w = self.window_infos[window_id]
        return self.x_full[w.train_window], self.y_full[w.train_window]

    @ functools.lru_cache
    def get_test_data(self,window_id):
        if window_id > len(self.windows):
            raise ValueError(f"Window ID {window_id} out of range for {len(self.windows)} windows.")
        w = self.window_infos[window_id]
        return self.x_full[w.train_test_window], self.y_full[w.train_test_window]

    @ functools.lru_cache
    def get_window_info(self,window_id):
        if window_id > len(self.windows):
            raise ValueError(f"Window ID {window_id} out of range for {len(self.windows)} windows.")
        return self.window_infos[window_id]

    def __iter__(self):
        for idx in range(self.n_windows):
            train_data = self.get_train_data(idx)
            test_data = self.get_test_data(idx)
            window_info = self.get_window_info(idx)
            yield idx, window_info,train_data, test_data

    def __len__(self):
        return self.n_windows
    
    def __repr__(self):
        return f"SeriesData(n_windows={self.n_windows}, kernel_width={params.kernel_width}, los_cutoff={params.los_cutoff})"
        
series_data = SeriesData(df_occupancy, params)


class SeriesFitResult:
    def __init__(self, distro):
        self.distro = distro
        self.window_infos = []
        self.fit_results = []
        self.train_relative_errors = None
        self.test_relative_errors = None        
        self.successes = []
        self.n_success = np.nan

    def append(self, window_info, fit_result):
        if not isinstance(window_info, WindowInfo):
            raise TypeError("window_info must be an instance of WindowInfo")
        if not isinstance(fit_result, SingleFitResult):
            raise TypeError("fit_result must be an instance of SingleFitResult")
        self.window_infos.append(window_info)
        self.fit_results.append(fit_result)

    def bake(self):
        self._collect_errors()
        self.successes = [fr.success  for fr in self.fit_results]
        self.n_success = sum(self.successes)
        self.transition_rates = np.array([fr.params[0] if (fr is not None) else np.nan for fr in self.fit_results  ])
        self.transition_delays = np.array([fr.params[1] if (fr is not None) else np.nan for fr in self.fit_results ])
        return self

    def _collect_errors(self):
        self.errors_collected = True
        train_err = np.empty(len(self.fit_results))
        test_err = np.empty(len(self.fit_results))
        for i, fr in enumerate(self.fit_results):
            if fr is None:
                train_err[i] = np.inf
                test_err[i] = np.inf
                continue
            train_err[i] = fr.rel_train_error
            test_err[i] = fr.rel_test_error
        self.train_relative_errors = train_err
        self.test_relative_errors = test_err
 
    def __getitem__(self, window_id):
        if isinstance(window_id, slice):
            return self.fit_results[window_id]
        if window_id >= len(self.fit_results):
            raise IndexError(f"Window ID {window_id} out of range for {len(self.fit_results)} windows.")
        return self.fit_results[window_id]

    def __setitem__(self, window_id, value):
        if window_id >= len(self.fit_results):
            raise IndexError(f"Window ID {window_id} out of range for {len(self.fit_results)} windows.")
        self.fit_results[window_id] = value
    
    def __repr__(self):
        return f"SeriesFitResult(distro={self.distro}, n_windows={len(self.window_infos)}, train_relative_error={self.train_relative_errors}, test_relative_error={self.test_relative_errors})"


class MultiSeriesFitResults(OrderedDict):
    def __init__(self, distros,*args, **kwargs):
        super().__init__(*args, **kwargs)
        for distro in distros:
            self[distro] = SeriesFitResult(distro)
        self.distros = list(self.keys())
        self.results = list(self.values())

    def bake(self):
        for distro, fit_result in self.items():
            fit_result.bake()
        self.n_windows = len(self.results[0].fit_results) if self.results else 0
        self.train_errors_by_distro = np.array([fr.train_relative_errors for fr in self.results]).T
        self.test_errors_by_distro = np.array([fr.test_relative_errors for fr in self.results]).T
        self.successes_by_distro = np.array([fr.successes for fr in self.results]).T
        self.failures_by_distro = 1 - self.successes_by_distro.astype(int)
        self.n_success_by_distro = np.array([fr.n_success for fr in self.results]).T
        self.transition_rates_by_distro = np.array([fr.transition_rates for fr in self.results]).T
        self.transition_delays_by_distro = np.array([fr.transition_delays for fr in self.results]).T
        self.n_windows = len(self.results[0].fit_results) if self.results else 0                                                                                                  
        
        self._make_summary()
        return self

    def _make_summary(self):
        df_train = pd.DataFrame(all_fit_results.train_errors_by_distro, columns=all_fit_results.distros)
        df_test = pd.DataFrame(all_fit_results.test_errors_by_distro, columns=all_fit_results.distros)

        # Compute mean finite loss and failure rate for each model
        summary = pd.DataFrame(index=all_fit_results.distros)
        summary["Failure Rate"] = all_fit_results.failures_by_distro.mean(axis=0)

        summary["Mean Loss Train"] = df_train.replace(np.inf, np.nan).mean()
        summary["Median Loss Train"] = df_train.replace(np.inf, np.nan).median()
        summary["Upper Quartile Train"] = df_train.quantile(0.75)
        summary["Lower Quartile Train"] = df_train.quantile(0.25)

        summary["Mean Loss Test"] = df_test.replace(np.inf, np.nan).mean()
        summary["Median Loss Test"] = df_test.replace(np.inf, np.nan).median()

        def remove_outliers(df, col):
            summary[col] = np.nan
            for distro in all_fit_results.distros:
                Q1,Q3 = df[distro].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                # filter out outliers
                mask = (df[distro] < (Q1 - 1.5 * IQR)) | (df[distro] > (Q3 + 1.5 * IQR))
                summary.at[distro,col] = df[distro][~mask].mean()

        remove_outliers(df_test,"Mean Loss Test (no outliers)")
        remove_outliers(df_train,"Mean Loss Train (no outliers)")
        
        self.summary = summary


        


    def __repr__(self):
        return f"MultiSeriesFitResults(distros={self.distros}, n_windows={self.n_windows})"

def get_initial_params(all_fit_results, distro, params, window_id):
    fit_result = all_fit_results[distro]
    if params.reuse_last_parametrization:
        for prev in reversed(fit_result[:window_id]):
            if not prev:
                continue
            return prev.params[2:]

    # fallback to df_init
    if distro in df_init.index:
        return df_init.loc[distro, "params"]
    return []

#%%

visualization_data.xlims = xlims
visualization_data.results_folder = results_folder
visualization_data.figures_folder = figures_folder
visualization_data.animation_folder = animation_folder

#%%
import types
from pathlib import Path

import numpy as np

from los_fitter import fit_SEIR, fit_kernel_to_series, SingleFitResult
from convolutional_model import calc_its_convolution
from compartmental_model import calc_its_comp

# --- Configuration flags (could come from argparse or a config object) ---
DEBUG_WINDOWS = False
DEBUG_DISTROS = False
ONLY_LINEAR = False
LESS_WINDOWS = False

# Distributions we explicitly skip
EXCLUDE_DISTROS = {"beta", "invgauss", "gamma", "weibull", "lognorm", "sentinel", "block"}

# --- Prepare input

x_full, y_full = select_series(df_occupancy, params)

# --- Build list of distros to fit ---
base_distros = [d for d in distributions if d not in EXCLUDE_DISTROS]
all_distros = base_distros + ["compartmental"]
if DEBUG_DISTROS:
    distro_to_fit = ["linear", "compartmental"]
elif ONLY_LINEAR:
    distro_to_fit = ["linear"]
else:
    distro_to_fit = [d for d in all_distros if d not in EXCLUDE_DISTROS]

# --- Window enumeration with optional debugging slicing ---
window_data = list(series_data)

if LESS_WINDOWS:
    window_data = window_data[:3]
elif DEBUG_WINDOWS:
    window_data = window_data[10:11]

# --- Prepare kernel storage ---
kernels_per_week = {
    d: (None if d == "compartmental" else np.zeros((len(x_full), params.kernel_width)))
    for d in distro_to_fit
}

all_fit_results = MultiSeriesFitResults(distro_to_fit)
trans_rates = []
delays = []


import tqdm
# --- Main loop ---

for distro in distro_to_fit:
    print(f"Distro: {distro}")
    failed_windows = []
    first_window = True
    # SEIR always uses its own fitter
    for window_id, window_info, train_data, test_data in tqdm.tqdm(window_data):
        w = window_info

        # Build curve_init and boundary tuples if needed
        curve_init, curve_bounds = None, None
        if params.fit_admissions:
            curve_init = [1, 0]
            curve_bounds = [(1, 1), (0, 0)]

        # per‐distro initialization
        init_vals = get_initial_params(all_fit_results, distro, params, window_id)
        distro_bounds = [(v, v) for v in init_vals]

        try:
            if distro == "compartmental":
                result_dict, result_obj = fit_SEIR(
                    *train_data,*test_data,
                    initial_guess_comp=[1/7, 1, 0],
                    los_cutoff=params.los_cutoff,
                )
                y_pred = calc_its_comp(x_full, *result_obj.params, y_full[0])
            else:
                past_k = None
                if not first_window and params.variable_kernels:
                    past_k = kernels_per_week[distro][w.train_start : w.train_start + params.los_cutoff]
                result_dict, result_obj = fit_kernel_to_series(
                    distro,
                    *train_data,*test_data,
                    params.kernel_width, params.los_cutoff,
                    curve_init, curve_bounds,
                    distro_init_params=init_vals,
                    past_kernels=past_k,
                    error_fun=params.error_fun,
                    fit_transition_rate=not params.fit_admissions,
                )
                # update kernel store
                kernel_full = kernels_per_week[distro]
                k = result_obj.kernel
                if first_window:
                    kernel_full[:] = k
                else:
                    kernel_full[w.train_start :] = k
                y_pred = calc_its_convolution(
                    x_full, kernel_full, *result_obj.params[:2], params.los_cutoff
                )

            # record transition & delay
            trans_rates.append(result_obj.params[1])
            delays.append(result_obj.params[0])

            # compute errors
            rel_err = np.abs(y_pred - y_full) / (y_full + 1)
            result_dict["train_relative_error"] = np.mean(rel_err[w.train_window])
            result_dict["test_relative_error"]  = np.mean(rel_err[w.test_window])
            result_obj.rel_train_error = np.mean(rel_err[w.train_window])
            result_obj.rel_test_error = np.mean(rel_err[w.test_window])

        except Exception as e:
            print(f"\tError fitting {distro}: {e}")
            dummy = types.SimpleNamespace(success=False)
            result_dict = {"minimization_result": dummy, "success": False}
            result_obj = SingleFitResult()


        result_dict["success"] = result_dict["minimization_result"].success
        if not result_dict["success"]:
            failed_windows.append(window_id)
        all_fit_results[distro].append(window_info,result_obj)

        first_window = False
    if failed_windows:
        print(f"Failed windows for {distro}: {failed_windows}")

all_fit_results.bake()

for distro, fit_result in all_fit_results.items():
    a = fit_result.train_relative_errors.mean()
    b = fit_result.test_relative_errors.mean()
    print(f"{distro[:7]}\t Mean Train Error: {float(a):.2f}, Mean Test Error: {float(b):.2f}")
  
#%% Calculate Summary

#%%
replace_names = {"block":"Constant Discharge","sentinel":"Baseline: Sentinel"}
replace_short_names =  {"exponential":"exp","gaussian":"gauss","compartmental":"comp"}
short_distro_names = [distro if distro not in replace_short_names else replace_short_names[distro] for distro in all_fit_results]

distro_colors = {distro: graph_colors[i] for i, distro in enumerate(all_fit_results)}
distro_patches = [
    Patch(color=distro_colors[distro], label=replace_names.get(distro, distro.capitalize()))
    for distro in all_fit_results
]

visualization_data.replace_names = replace_names
visualization_data.distro_colors = distro_colors
visualization_data.short_distro_names = short_distro_names
visualization_data.distro_patches = distro_patches

#%% Generate Video 
##animation
# pair each distribution with a color from graph_colors

DEBUG_ANIMATION = True
DEBUG_HIDE_FAILED = True

visualize_fit_deconvolution(
        all_fit_results,
        series_data,
        params,
        vd,
        df_mutant_selection,
        DEBUG_ANIMATION,
        DEBUG_HIDE_FAILED,
        )

#%%
# Run models on a pulse

run_pulse_model(params.run_name, vd.animation_folder, all_fit_results, series_data, debug=True)

#%%

# Just fit SEIR-Model
from los_fitter import fit_SEIR

initial_guess_comp = [1/7,0.02,0]


fig,ax = plt.subplots(1,1,figsize=(10,7),sharex=True)
ax.plot(y_full,label="Real",color="black")
for window_id, window_info, train_data, test_data in tqdm.tqdm(window_data):
    w = window_info
    x_train, y_train = train_data
    x_test, y_test = test_data

    result_dict,result_obj = fit_SEIR(*train_data,*test_data, initial_guess_comp,
                      los_cutoff=params.los_cutoff)
    y_pred_b = calc_its_comp(x_test, *result_obj.params,y_test[0])
    xs2 = np.arange(w.train_end,w.test_end)
    ax.plot(xs2,y_pred_b[params.train_width:],color=graph_colors[1])
ax.set_xlim(*vd.xlims)
plt.title("SEIR-Models")
plt.show()
#%%
# Unload high_vizzz DeconvolutionPlots
del DeconvolutionPlots

#%%
from high_vizzz import DeconvolutionPlots

visualizer = DeconvolutionPlots(all_fit_results,series_data,params,vd)
visualizer.generate_plots_for_run()

#%%
message = f"Finished Run - {params.run_name}"
print(message)
os.system(f'msg * "{message}"')


# %%
