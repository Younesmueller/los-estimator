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
manual_transition_rates = get_manual_transition_rates(df_occupancy)

plt.plot(manual_transition_rates,label="Manual Transition Rates")
plt.title("Manual Transition Rates")
plt.xticks(xtick_pos[::4],xtick_label[::4])
show_plt()


#%%

axs = df_occupancy.plot(subplots=True)
for ax in axs:
    ax.axvspan(sentinel_start_date,sentinel_end_date, color="green", alpha=0.1,label="Sentinel")
axs[-1].axvline(new_icu_date,color="black",linestyle="--",label="First ICU")

plt.suptitle("Incidences and ICU Occupancy")
show_plt()

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

params = types.SimpleNamespace()
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

print("###################################################################")
print("Run Name:")
print(run_name)
print("###################################################################")


results_folder, figures_folder, animation_folder = create_result_folders(run_name)
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
class WindowInfo:
    def __init__(self,window):
        self.window = window
        self.train_end = self.window
        self.train_start = self.window - params.train_width
        self.train_los_cutoff = self.train_start + params.los_cutoff
        self.test_start = self.train_end
        self.test_end = self.test_start + params.test_width

        self.train_window = slice(self.train_start,self.train_end)
        self.train_test_window = slice(self.train_start,self.test_end)
        self.test_window = slice(self.test_start,self.test_end)
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
        self.window_infos = [WindowInfo(window) for window in self.windows]
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

for distro, fit_result in all_fit_results.items():
    fit_result.bake()        
    a = fit_result.train_relative_errors.mean()
    b = fit_result.test_relative_errors.mean()
    print(f"{distro[:7]}\t Mean Train Error: {float(a):.2f}, Mean Test Error: {float(b):.2f}")
#%% Calculate Summary
train_errors_by_distro = np.array([result.train_relative_errors for result in all_fit_results.results]).T
df_train = pd.DataFrame(train_errors_by_distro, columns=all_fit_results.distros)

test_errors_by_distro = np.array([result.test_relative_errors for result in all_fit_results.results]).T
df_test = pd.DataFrame(test_errors_by_distro, columns=all_fit_results.distros)

fails = 1 - np.array([result.successes for result in all_fit_results.results])
df_failures = pd.DataFrame(fails.T, columns=all_fit_results.distros)
# Compute mean finite loss and failure rate for each model
summary = pd.DataFrame(index=all_fit_results.distros)
summary["Mean Loss Train"] = df_train.replace(np.inf, np.nan).mean()
summary["Median Loss Train"] = df_train.replace(np.inf, np.nan).median()
summary["Failure Rate Train"] = df_failures.mean()
summary["Upper Quartile Train"] = df_train.quantile(0.75)
summary["Lower Quartile Train"] = df_train.quantile(0.25)

summary["Mean Loss Test"] = df_test.replace(np.inf, np.nan).mean()
summary["Median Loss Test"] = df_test.replace(np.inf, np.nan).median()
summary["Failure Rate Test"] = df_failures.mean()
# add column for mean loss without oultiers
col = "Mean Loss Test (no outliers)"
summary[col] = np.nan

# Find outliers
for distro in all_fit_results.distros:
    Q1 = df_test[distro].quantile(0.25)
    Q3 = df_test[distro].quantile(0.75)
    IQR = Q3 - Q1
    # filter out outliers
    mask = (df_test[distro] < (Q1 - 1.5 * IQR)) | (df_test[distro] > (Q3 + 1.5 * IQR))
    summary.at[distro,col] = df_test[distro][~mask].mean()
summary

#%%
replace_names = {"block":"Constant Discharge","sentinel":"Baseline: Sentinel"}
replace_short_names =  {"exponential":"exp","gaussian":"gauss","compartmental":"comp"}
short_distro_names = [distro if distro not in replace_short_names else replace_short_names[distro] for distro in all_fit_results]

distro_colors = {distro: graph_colors[i] for i, distro in enumerate(all_fit_results)}
distro_patches = [
    Patch(color=distro_colors[distro], label=replace_names.get(distro, distro.capitalize()))
    for distro in all_fit_results
]


#%%
# save test errors and failure rates in csv
df_test.to_csv(results_folder + "test_errors.csv")
df_failures.to_csv(results_folder + f"failure_rates.csv")
summary.to_csv(results_folder + "summary.csv")


#%% Generate Video 
##animation
# pair each distribution with a color from graph_colors


debug = True
show_mutants = True
hide_failed = True
if True:
    if not debug:
        path = animation_folder
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path)

    window_counter = 1
    to_enumerate = list(enumerate(series_data.window_infos))
    if debug:
        xx = 2
        to_enumerate = [to_enumerate[min(xx,len(to_enumerate)-1)]]
    for window_id, window_info in to_enumerate:
        print(f"Animation Window {window_counter}/{len(windows)}")
        window_counter+=1


        if show_mutants:
            fig = plt.figure(figsize=(17, 10),dpi=150)
            gs = gridspec.GridSpec(3, 4,height_ratios=[5,1,3])
            ax_main = fig.add_subplot(gs[0, :4])
            ax_inc = ax_main.twinx()
            ax_kernel = fig.add_subplot(gs[2,:2])
            ax_err_train = fig.add_subplot(gs[2, 2])
            ax_err_test = fig.add_subplot(gs[2, 3])
            ax_mutant = fig.add_subplot(gs[1, :4])
        else:
            fig = plt.figure(figsize=(17, 10),dpi=150)
            gs = gridspec.GridSpec(2, 4,height_ratios=[2,1])
            ax_main = fig.add_subplot(gs[0, :4])
            ax_inc = ax_main.twinx()
            ax_kernel = fig.add_subplot(gs[1,:2])
            ax_err_train = fig.add_subplot(gs[1, 2])
            ax_err_test = fig.add_subplot(gs[1, 3])

        w = window_info

        # Plot main window
        line_bedload, = ax_main.plot(y_full, color="black",label="ICU Bedload")
        zero = np.zeros_like(y_full)

        span_los_cutoff = ax_main.axvspan(w.train_start, w.train_los_cutoff, color="magenta", alpha=0.1,label=f"Train Window (Convolution Edge) = {params.train_width} days")
        span_train = ax_main.axvspan(w.train_los_cutoff, w.train_end, color="red", alpha=0.2,label=f"Training = {params.train_width-params.los_cutoff} days")
        span_test = ax_main.axvspan(w.test_start, w.test_end, color="blue", alpha=0.05,label=f"Test Window = {params.test_width} days")
        ax_main.axvline(w.train_end,color="black",linestyle="-",linewidth=1)

        label = "COVID Incidence (Scaled)"
        if params.fit_admissions:
            label = "New ICU Admissions (Scaled)"


        line_inc, = ax_inc.plot(x_full,linestyle="--",label=label)
        ax_inc.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        ma = np.nanmax(x_full)
        ax_inc.set_ylim(-ma/7.5,ma*4)

        plot_lines  = []

        for distro,result_series in all_fit_results.items():
            result_obj = result_series.fit_results[window_id]
            name = replace_names.get(distro,distro.capitalize())
            if hide_failed and not result_obj.success:
                continue
            c = distro_colors[distro]
            ax_kernel.plot(result_obj.kernel, label=name, color=c)
            y = result_obj.curve[params.los_cutoff:]
            s = np.arange(len(y))+params.los_cutoff+w.train_start
            l, = ax_main.plot(s,y, label=f"{distro.capitalize()}", color=c)
            plot_lines.append(l)

        ax_kernel.plot(real_los, color="black",label="Sentinel LoS Charité")

        for i, (distro, fit_result) in enumerate(all_fit_results.items()):
            c = distro_colors[distro]
            train_err = fit_result.train_relative_errors[window_id]
            test_err = fit_result.test_relative_errors[window_id]

            if hide_failed and not fit_result[window_id].success:
                ax_err_train.bar(i,1e100,color="lightgrey",hatch="/")
                ax_err_test.bar(i, 1e100,color="lightgrey",hatch="/")
                ax_err_train.bar(i,train_err,color="black")
                ax_err_test.bar(i,test_err,color="black")
                continue
            ax_err_train.bar(i,train_err,label="Train",color=c)
            ax_err_test.bar(i,test_err,label="Test",color=c)

        legend1 = ax_main.legend(handles=distro_patches, loc="upper left", fancybox=True,ncol=2)
        legend2 = ax_main.legend(handles = [line_bedload, line_inc, span_los_cutoff, span_train, span_test],loc="upper right")

        ax_main.add_artist(legend1)
        ax_main.add_artist(legend2)

        ax_main.set_title(f"ICU Occupancy")
        ax_main.set_xticks(xtick_pos)
        ax_main.set_xticklabels(xtick_label)
        ax_main.set_xlim(*xlims)
        ax_main.set_ylim(-200,6000)
        ax_main.set_ylabel("Occupied Beds")

        if show_mutants:

            mutant_lines = []
            for col in df_mutant_selection.columns:
                line, = ax_mutant.plot(df_mutant_selection[col].values)
                mutant_lines.append(line)
                ax_mutant.fill_between(range(len(y_full)), df_mutant_selection[col].values, 0, alpha=0.3)

            ax_mutant.legend(mutant_lines,df_mutant_selection.columns,loc="upper right")
            ax_mutant.set_xticks([])
            ax_mutant.set_xticklabels([])
            ax_mutant.set_xticks(xtick_pos)
            tmp_xtick = [label.split("\n")[1:] for label in xtick_label]
            tmp_xtick = [label[0] if label else ""  for label in tmp_xtick]
            ax_mutant.set_xticklabels(tmp_xtick)
            ax_mutant.set_xlim(*xlims)
            ax_mutant.set_title("Variants of Concern")
            ax_mutant.set_ylabel("Variant Share (%)")




        ax_inc.set_ylabel("(Incidence)")
        if params.fit_admissions:
            ax_inc.set_ylabel("New ICU Admissions (scaled)")
        replace_names = {"block": "Constant Discharge", "sentinel": "Baseline: Sentinel"}
        ax_kernel.legend(handles=distro_patches, loc="upper right", fancybox=True, ncol=2, )
        ax_kernel.set_ylim(0,0.1)
        ax_kernel.set_xlim(-2,80)
        ax_kernel.set_ylabel("Discharge Probability")
        ax_kernel.set_xlabel("Days after admission")
        ax_kernel.set_title(f"Estimated LoS Kernels")


        lim = .4
        ax_err_train.set_ylim(0,lim)
        ax_err_train.set_title("Relative Train Error")
        ax_err_train.set_xticks(range(len(all_fit_results)))
        ax_err_train.set_xticklabels(short_distro_names,rotation=75)
        ax_err_train.set_ylabel("Relative Error")

        ax_err_test.set_ylim(0,lim)
        ax_err_test.set_title("Relative Test Error")
        ax_err_test.set_xticks(range(len(all_fit_results)))
        ax_err_test.set_xticklabels(short_distro_names,rotation=75)
        ax_err_test.set_ylabel("Relative Error")

        plt.suptitle(f"Deconvolution Training Process\n{run_name.replace('_',' ')}",fontsize=16)
        plt.tight_layout()
        if debug:
            show_plt()
        else:
            plt.savefig(animation_folder + f"fit_{w.train_start:04d}.png")
            plt.close()
        plt.clf()



#%%
# Run models on a pulse


run_pulse_model(run_name, animation_folder, all_fit_results, series_data, debug=True)

#%%

# Just fit SEIR-Model
from los_fitter import fit_SEIR
from scipy.optimize import minimize

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
ax.set_xlim(*xlims)
plt.title("SEIR-Models")
plt.show()
#%%
# Plot number of failed fits and successfull fits
plt.figure(figsize=(10,5),dpi=150)
for i,distro in enumerate(all_fit_results):
    plt.bar(i,all_fit_results[distro].n_success,color=graph_colors[i],label=distro.capitalize())
plt.xticks(np.arange(len(all_fit_results)),all_fit_results.keys(),rotation=45)
plt.title("Number of successful fits")
plt.axhline(series_data.n_days,color="red",linestyle="--",label="Total")
plt.xticks(rotation=45)
plt.savefig(figures_folder + "successful_fits.png")

show_plt()

#%%
#%%

# Visualization
def viz(col2, col1,ylim=None,save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6),dpi=150)

    for i, distro in enumerate(all_fit_results.distros):
        if distro in ["sentinel","block"]:
            continue
        val1 = summary[col1][distro]
        val2 = summary[col2][distro]
        ax.scatter(val1, val2, s=100, label=distro, color=graph_colors[i])
        ax.annotate(distro, (val1, val2), fontsize=9, xytext=(5,5), textcoords='offset points')

    # Labels and formatting
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"Model Performance: {col1} vs. {col2}\n{run_name.replace('_',' ')}")
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path)
    show_plt()
viz("Median Loss Train", "Failure Rate Train",          save_path=figures_folder +  "median_loss_vs_failure_rate_train.png")
viz("Median Loss Test", "Failure Rate Test",save_path=figures_folder +  "median_loss_vs_failure_rate_test.png")
col = "Mean Loss Test (no outliers)"
viz(col, "Failure Rate Test",save_path=figures_folder +  "mean_loss_vs_failure_rate_test.png")
#%%
# sorted summary
sorted_summary = summary.sort_values("Median Loss Test")
sorted_summary = sorted_summary[["Median Loss Test","Failure Rate Train","Median Loss Train","Upper Quartile Train","Lower Quartile Train"]]
sorted_summary.plot(subplots=True,figsize=(10,10))
plt.legend()
plt.title("Median Loss")
xticks = list(sorted_summary.index)
plt.xticks(np.arange(len(xticks)),xticks,rotation=45)
show_plt()
#%%

plt.figure(figsize=(10,5),dpi=150)
plt.boxplot(train_errors_by_distro)
distro_and_n = [f"{distro.capitalize()} n={fr.n_success}" for distro, fr in all_fit_results.items()]
plt.xticks(np.arange(len(distro_and_n))+1,distro_and_n,rotation=45)
plt.title("Train Error")
plt.ylabel("Relative Train Error")
plt.tight_layout()
plt.savefig(figures_folder + "train_error_boxplot.png")
show_plt()

plt.figure(figsize=(10,5),dpi=150)
plt.boxplot(test_errors_by_distro)
plt.xticks(np.arange(len(distro_and_n))+1,distro_and_n,rotation=45)
plt.title(f"Test Error\n{run_name}")
plt.ylabel(f"Relative Error")
plt.tight_layout()
plt.savefig(figures_folder + "test_error_boxplot.png")
show_plt()
#%%
fig = plt.figure(figsize=(10,5))
sns.stripplot(data=train_errors_by_distro, jitter=0.2)
plt.xticks(np.arange(len(all_fit_results)),all_fit_results.distros,rotation=45)
plt.title(f"Train Error\n{run_name}")
plt.savefig(figures_folder + "train_error_stripplot.png")
show_plt()




#%%
for distro_id, distro in enumerate(all_fit_results.distros):
    fr_series = all_fit_results[distro]
    fig,(ax,ax4,ax2)= plt.subplots(3,1,figsize=(10,5),sharex=True,dpi=150)
    ax.plot(y_full, color="black",label="Real" ,alpha=.8,linestyle="--")
    for w,fit_result in zip(fr_series.window_infos,fr_series.fit_results):
        
        if not fit_result.success:
            ax.axvspan(w.train_start,w.train_end, color="red", alpha=0.1)
            continue

        y = fit_result.curve[params.los_cutoff:params.train_width]
        x = np.arange(w.train_los_cutoff,w.train_end)
        l1, = ax.plot(x, y, color=graph_colors[0])

        y = fit_result.curve[params.train_width:params.train_width+params.test_width]
        x = np.arange(w.train_end,w.test_end)
        l2, = ax.plot(x, y, color=graph_colors[1])


    ax.plot([],[],color=graph_colors[0],label = f"{distro.capitalize()} Train")
    ax.plot([],[],color=graph_colors[1],label = f"{distro.capitalize()} Prediction")
    ax.axvspan(0,0, color="red", alpha=0.1,label="Failed Training Windows")
    ax.axvspan(sentinel_start_day,sentinel_end_day, color="green", alpha=0.1,label="Sentinel Window")
    ax.legend(loc="upper left")
    ax.set_ylim(-100,6000)
    ax.set_xticks(xtick_pos[::2])
    ax.set_xticklabels(xtick_label[::2])
    ax.set_xlim(*xlims)
    ax.grid()

    # plot trans rates in ax2
    x = series_data.windows
    ax2.bar(x, all_fit_results[distro].transition_rates, width=15 ,label="Transition Probability")
    ax2.grid()
    ax2.set_ylim(-.01,0.1)
    ax2.set_title("Transition Probability")
    ax4.plot(x, all_fit_results[distro].train_relative_errors,label = "Train Error")
    ax4.plot(x, all_fit_results[distro].test_relative_errors,label = "Test Error")
    # mark nan and inf values
    fit_series = all_fit_results[distro]
    for i, fit_result in enumerate(fit_series.fit_results):
        if not fit_result.success:
            ax4.axvline(x[i],color="red",alpha=.5)
    ax4.axvline(-np.inf,color="red",label="Failed Fit",alpha=.5)
    # ax4.axvspan(sentinel_start_day,sentinel_end_day, color="green", alpha=0.1,label="Sentinel Window")
    ax4.legend(loc="upper right")
    ax4.set_ylim(-1e4,1e5)
    ax4.set_title("Error")
    ax4.grid()

    plt.suptitle(f"{distro.capitalize()} Distribution\n{run_name}")
    plt.tight_layout()
    plt.savefig(figures_folder + f"prediction_error_{distro}_fit.png")
    show_plt()
#%%
fig,(ax,ax4)= plt.subplots(2,1,figsize=(12,6),sharex=True,dpi=300)
ax.plot(y_full, color="black",label="Real" ,alpha=.8,linestyle="--")
ax.plot([],[],color=graph_colors[0],label = f"{distro.capitalize()} Train")
ax.plot([],[],color=graph_colors[1],label = f"{distro.capitalize()} Prediction")
ax.axvspan(0,0, color="red", alpha=0.1,label="Failed Training Windows")

for distro_id, distro in enumerate(all_fit_results.distros):
    fr_series = all_fit_results[distro]
    for w,fit_result in zip(fr_series.window_infos,fr_series.fit_results):
        
        if not fit_result.success:
            ax.axvspan(w.train_start,w.train_end, color="red", alpha=0.1)
            continue

        y = fit_result.curve[params.los_cutoff:params.train_width]
        x = np.arange(w.train_los_cutoff,w.train_end)
        l1, = ax.plot(x, y, color=graph_colors[0])

        y = fit_result.curve[params.train_width:params.train_width+params.test_width]
        x = np.arange(w.train_end,w.test_end)
        l2, = ax.plot(x, y, color=graph_colors[1])

    ax4.plot(series_data.windows,train_errors_by_distro.T[distro_id],color=graph_colors[0])
    ax4.plot(series_data.windows,test_errors_by_distro.T[distro_id], color=graph_colors[1])

ax.set_ylim(-100,6000)
if params.fit_admissions:
    ax.set_xlim(*xlims)
ax.grid()
ax.legend()
ax4.axvline(-np.inf,color="red",label="Failed Fit",alpha=.5)
ax4.legend(["Train Errors","Test Errors"],loc="upper right")
ax4.set_ylim(-1e4,1e5)
ax4.grid()
# ax4.set_xticks(xtick_pos,xtick_label)

ax4.set_xlabel("Time")
ax4.set_ylabel("Error")
ax.set_ylabel("ICU")
# set xticks
ax.set_xticks(xtick_pos[::2])
ax.set_xticklabels(xtick_label[::2])
ax.set_xlim(*xlims)
ax.set_title(f"All Predictions\n{run_name}")
plt.savefig(figures_folder + "prediction_error_all_distros.png")
show_plt()

#%%
fig,ax= plt.subplots(1,1,figsize=(15,7.5),sharex=True)
for distro_id, distro in enumerate(all_fit_results.distros):
    fr_series = all_fit_results[distro]
    for w,fit_result in zip(fr_series.window_infos,fr_series.fit_results):
        
        if not fit_result.success:
            continue

        y = fit_result.curve[params.los_cutoff:params.train_width]
        x = np.arange(w.train_los_cutoff,w.train_end)
        l1, = ax.plot(x, y, color=graph_colors[0])

        y = fit_result.curve[params.train_width:params.train_width+params.test_width]
        x = np.arange(w.train_end,w.test_end)
        l2, = ax.plot(x, y, color=graph_colors[1])

    ax4.plot(series_data.windows,train_errors_by_distro.T[distro_id],color=graph_colors[0])
    ax4.plot(series_data.windows,test_errors_by_distro.T[distro_id], color=graph_colors[1])

ax.plot(y_full, color="black",label="Real" ,alpha=.8,linestyle="--")

ax.set_ylim(-100,6000)
ax.set_xlim(*xlims)
ax.grid()

ax.plot([],[],color=graph_colors[0],label = f"All Distros Train")
ax.plot([],[],color=graph_colors[1],label = f"All Distros Prediction")
ax.legend()
ax.set_title(f"All Models\n{run_name}")
plt.savefig(figures_folder + "prediction_error_all_fits.png")
show_plt()


# %%
# plot distributions

for distro, fit_results in all_fit_results.items():
    fig, ax = plt.subplots(figsize=(10,5))
    # plot real kernel
    ax.plot(real_los,color='black',label="Real")

    for fit_result in fit_results.fit_results:
        if not fit_result.success:
            continue
        y = fit_result.kernel
        ax.plot(y,alpha=0.3,color=graph_colors[0])

    plt.grid()
    plt.legend()
    plt.title(f"{distro.capitalize()} Kernel\n{run_name}")
    plt.ylim(-0.005,0.3)
    plt.tight_layout()
    plt.savefig(figures_folder + f"all_kernels_{distro}.png")
    show_plt()
#%%



def calculate_manual_transition_rates_from_sentinel_distro(show_plt, df_occupancy, windows, all_fit_results):
    if "sentinel" not in all_fit_results.distros:
        return

    from scipy.optimize import minimize
    from matplotlib.patches import Patch

    # Fit transition rates manually
    tr = all_fit_results["linear"].transition_rates
    tr = np.insert(tr,0,tr[0])
    mps = np.array([
        [0,0.001],
        [48,0.001],
        [69,0.001],
        [111,0.05146966],
        [174,0.05982728],
        [237,0.01350479],
        [500,0.01350479],
        [594,0.01850925],
        [783,0.00042575],
        [1119,0.0017396 ],
        [1329,0.02537781],
        ])
    xs = np.arange(len(df_occupancy))
    windows2 = np.insert(windows.copy(),0,0)
    tr_y = np.interp(xs,windows2,tr)

    def obj_fun(ps):
        ps = ps.reshape(-1,2)
        ys = np.interp(xs,ps[:,0],ps[:,1])
        return np.mean((ys-tr_y)**2)
    result = minimize(
        obj_fun,
        mps.flatten(),
        method = "L-BFGS-B"
        )
    print(result)

    fig,ax = plt.subplots(1,1,figsize=(10,5))
    res = result.x.reshape(-1,2)
    ys = np.interp(xs,res[:,0],res[:,1])
    plt.plot(ys,label="Fitted Transition Rates")
    plt.plot(tr_y,label="Real Transition Rates")
    plt.plot(np.abs(ys-tr_y),label="Difference")
    plt.grid()
    plt.legend()
    show_plt()
    manual_points = np.array(
        [[0,               5.26125540e-02],
        [ 4.79999995e+01,  5.29817366e-02],
        [ 6.89999731e+01,  5.25110807e-02],
        [ 1.10999986e+02,  5.42751008e-02],
        [ 1.73999994e+02,  3.54917868e-02],
        [ 2.37000003e+02,  1.42336509e-02],
        [ 4.99999999e+02,  1.48902533e-02],
        [ 5.94000000e+02,  1.08504095e-02],
        [ 7.83000001e+02,  1.49297704e-03],
        [ 1.11900000e+03,  8.93606945e-04],
        [ 1.32900000e+03,  1.76263698e-02]]
        )
    xs = np.arange(len(df_occupancy))
    ys = np.interp(xs,manual_points[:,0],manual_points[:,1])

    print(obj_fun(result.x))
    print(result.x.reshape(-1,2))

calculate_manual_transition_rates_from_sentinel_distro(show_plt, df_occupancy, series_data.windows, all_fit_results)

# %%



# transition rates in a subplot together with the icu
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,10),sharex=True)
# ax1.plot(df["icu"].values,label="ICU")
y = df_occupancy["icu"].values/df_occupancy["AnzahlFall"].values
ax1.plot(y,label="icu incidence ratio")
ax1.set_title("ICU Occupancy")
ax1.legend()
for i,distro in enumerate(all_fit_results.distros):
    ax2.plot(series_data.windows, all_fit_results[distro].transition_rates,label=distro)
ax2.axhline(0,color="black",linestyle="--")
ax2.legend()
ax2.set_title("Transition Rates")
# ax2.set_ylim(-.03,0.2)
plt.savefig(figures_folder + "transition_rates_and_icu.png")
show_plt()
#%%

all_predictions_combined = np.empty((series_data.n_windows,series_data.n_days,len(all_fit_results)),dtype=np.float32)
for distro_count,result_series in enumerate(all_fit_results.results):
    for window_id, fit_result in enumerate(result_series.fit_results):
        y = fit_result.curve[params.train_width:params.train_width+params.test_width]
        w = result_series.window_infos[window_id]
        all_predictions_combined[window_id, w.train_end:w.test_end, distro_count] = y
#%%

fig,(ax,ax2) = plt.subplots(2,1,figsize=(10,5),sharex=True,dpi=150)
# plot icu and incidence
ax.plot(y_full, color="black",label="ICU Bedload")
for i, window in enumerate(series_data.windows):
    w = WindowInfo(window)
    x = np.arange(w.train_end,w.test_end)
    for j, distro in enumerate(all_fit_results.distros):
        y = all_predictions_combined[i,w.train_end:w.test_end,j]
        ax.plot(x,y,color=graph_colors[j],alpha=1)
ax.set_ylim(-100,6000)
ax.set_title("All Predictions")
ax.legend()
transition_rates = np.array([[fit_result.params[0] for fit_result in all_fit_results[distro]] for distro in all_fit_results.distros])
lines =[]
for i,distro in enumerate(all_fit_results.distros):
    l, = ax2.plot(series_data.windows, transition_rates[i],label=distro)
    lines.append(l)
ax2.legend(handles=lines,ncol=2,loc="upper right")
ax2.set_xticks(xtick_pos[::2])
ax2.set_xticklabels(xtick_label[::2])
# ax2.set_ylim(-.03,1.03)
ax2.set_ylim(-.01,.075)
ax2.set_xlim(50,1300)
ax2.set_title("Transition Rates")

plt.tight_layout()
plt.savefig(figures_folder + "transition_rates.png")
show_plt()
#%%
# plot all errors in a graph
fig,ax = plt.subplots(1,1,figsize=(10,5),dpi=150)
for i, distro in enumerate(all_fit_results):
    ax.plot(series_data.windows,all_fit_results[distro].train_relative_errors,label=f"{distro.capitalize()} Train")
    ax.plot(series_data.windows,all_fit_results[distro].test_relative_errors,label=f"{distro.capitalize()} Test")
ax.legend()
ax.set_title("All Errors")
ax.set_xticks(xtick_pos[::2])
ax.set_xticklabels(xtick_label[::2])
plt.grid()

#%%
message = f"Finished Run - {run_name}"
print(message)
os.system(f'msg * "{message}"')


# %%
