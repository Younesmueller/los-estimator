#%%
# reload imports
%load_ext autoreload
%autoreload 2
import os
import sys
import numpy as np
import pandas as pd
import types
import seaborn as sns

import time
from numba import njit
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.append("../02_fit_los_distributions/")
from dataprep import load_los, load_incidences, load_icu_occupancy, load_mutant_distribution
from compartmental_model import objective_function_compartmental,calc_its_comp
from los_fitter import fit_kernel_distro_to_data,fit_kernel_distro_to_data_with_previous_results, distributions, calc_its_convolution, fit_SEIR, generate_kernel
plt.rcParams['savefig.facecolor']='white'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
print("Let's Go!")
#%%
print_figs = True
def show_plt(*args,**kwargs):
    # if in vscode interactive
    if print_figs:
        plt.show()
    else:
        plt.clf()

#%%
los_file = "../01_create_los_profiles/berlin/output_los/los_berlin_all.csv"

kernel_width = 120
los_cutoff = 60 # Ca. 90% of all patients are discharged after 41 days

start_day = "2020-01-01"
end_day = "2025-01-01"

def date_to_day(date):
    return (date - pd.Timestamp(start_day)).days
def day_to_date(day):
    return pd.Timestamp(start_day) + pd.Timedelta(days=day)

#%%
# take matplotlib standart color wheel
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# add extra color palette
colors += ["#FFA07A","#20B2AA","#FF6347","#808000","#FF00FF","#FFD700","#00FF00","#00FFFF","#0000FF","#8A2BE2"]

#%%
real_los, _ = load_los(file=los_file)
df_inc, raw = load_incidences(start_day, end_day)
# wenn die ersten beiden Yeilen der 1.1.2020 sind entferne eine
if df_inc.index[0] == pd.Timestamp("2020-01-01") and df_inc.index[1] == pd.Timestamp("2020-01-01"):
    df_inc = df_inc.iloc[1:]

df_icu = load_icu_occupancy(start_day, end_day)
df = df_inc.join(df_icu,how="inner")
df["new_icu_smooth"] = df["new_icu"].rolling(7).mean()

sentinel_start_date =pd.Timestamp("2020-10-01")
sentinel_end_date = pd.Timestamp("2021-06-21")
sentinel_start_day = date_to_day(sentinel_start_date)
sentinel_end_day = date_to_day(sentinel_end_date)

new_icu_date = df.index[df["new_icu"]>0][0]
new_icu_day = date_to_day(new_icu_date)

axs = df.plot(subplots=True)
for ax in axs:
    ax.axvspan(sentinel_start_date,sentinel_end_date, color="green", alpha=0.1,label="Sentinel")
axs[-1].axvline(new_icu_date,color="black",linestyle="--",label="First ICU")

plt.suptitle("Incidences and ICU Occupancy")
show_plt()

#%%
xtick_pos = []
xtick_label = []
for i in range(0,len(df)):
    # take only first of month
    if df.index[i].day == 1:
        xtick_pos.append(i)
        # append month name
        label = df.index[i].strftime("%b")
        if i == 0 or df.index[i].month == 1:
            label += f"\n{df.index[i].year}"
        xtick_label.append(label)


#%%
#%%
fig,axs =  plt.subplots(3,1,figsize=(10,5),sharex=True)
(ax1,ax2,ax3) = axs
ax1.plot(df["icu"].values,                 label="Real ICU Bedload")
ax2.plot(df["AnzahlFall"].values,          label="Real Incidence")
ax3.plot(df["icu"].values/df["AnzahlFall"].values,label="Real ICU Bedload / Incidence")
ax1.set_title("ICU Bedload")
ax2.set_title("Incidence")
ax3.set_title("ICU Bedload / Incidence")
for ax in axs:
    ax.grid()
    ax.legend()
ax1.set_xticks(xtick_pos)
ax1.set_xticklabels(xtick_label)
ax1.set_xlim(140,270)
ax1.set_ylim(-100,500)
ax2.set_ylim(-100,2000)

plt.show()
#%%
fig,ax = plt.subplots(2,1,figsize=(10,5),sharex=True)

df["new_icu_smooth"].plot(ax=ax[1],label="new_icu",color="orange")
df["icu"].plot(ax=ax[0],label="AnzahlFall")
ax[0].set_title("Tägliche Neuzugänge ICU, geglättet")
ax[1].set_title("ICU Bettenbelegung")
plt.tight_layout()

plt.plot()


#%%
# Load init parameters
file = "../02_fit_los_distributions/output_los/los_berlin_all/fit_results.csv"
df_init = pd.read_csv(file,index_col=0)
df_init = df_init.set_index("distro")
# interpret params as array float of format [f1 f2 f3 ...]
df_init["params"] = df_init["params"].apply(lambda x: [float(i) for i in x[1:-1].split()])
df_init

#%%

df_mutant = load_mutant_distribution()
df_mutant = df_mutant.reindex(df.index,method="nearest")
df_mutant.plot()
df_mutant

#%%




params = types.SimpleNamespace()
params.use_manual_transition_rate = False
params.smooth_data = False
params.train_width = 42 + los_cutoff
params.test_width = 21 #28 *4
params.step = 7
params.fit_admissions = True
params.error_fun = "mse"# "weighted_mse"
params.reuse_last_parametrization = False
#%%
#%%

# try:
#     simulation_counter
# except:
#     simulation_counter = 2
# if simulation_counter==0:
#     params.error_fun="mse"
#     params.train_width = 7 + los_cutoff
# elif simulation_counter==1:
#     params.error_fun="mse"
#     params.train_width = 21 + los_cutoff
# elif simulation_counter==2:
#     params.error_fun="weighted_mse"
#     params.train_width = 7 + los_cutoff
# elif simulation_counter==3:
#     params.error_fun="weighted_mse"
#     params.train_width = 21 + los_cutoff
# else:
#     os.system(f'msg * "Finished Runs"')
#     raise Exception()
# simulation_counter+=1
# simulation_counter
#%%
timestamp = time.strftime("%y%m%d_%H%M")
run_name = f"{timestamp}_dev"

# for grid_search_id in range(0,12):

    # run_name = f"{timestamp}_grid_search"

    # params = types.SimpleNamespace()
    # # generate params from run_id
    # params.train_width = [7 + los_cutoff, 14 + los_cutoff, 21 + los_cutoff]
    # params.test_width = [7]
    # params.step = [21]
    # params.use_manual_transition_rate = [False, True]
    # params.fit_admissions = [True]

    # param_space = list(itertools.product(
    #     *params.__dict__.values()
    # ))
    # # set params dict values
    # for i,key in enumerate(params.__dict__.keys()):
    #     params.__dict__[key] = param_space[grid_search_id][i]


run_name+=f"_step{params.step}_train{params.train_width}_test{params.test_width}"
if params.use_manual_transition_rate:
    run_name += "_manual_transition"
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

print(run_name)
results_folder = f"./results/{run_name}/"
figures_folder = results_folder + "/figures/"
animation_folder = results_folder + "/animation/"

if os.path.exists(results_folder):
    import shutil
    shutil.rmtree(results_folder)    
os.makedirs(results_folder)
os.makedirs(figures_folder)
os.makedirs(animation_folder)
#%%
start = 0
if params.fit_admissions:
    start = new_icu_day+los_cutoff+21
windows = np.arange(start,len(df)-params.step,params.step)
# # remove windows, where y_test is too short
windows = np.array([window for window in windows if window+params.train_width+params.test_width < len(df)])
#%%
class WindowInfo:
    def __init__(self,window):
        self.window = window
        self.train_end = self.window
        self.train_start = self.window - params.train_width
        self.train_los_cutoff = self.train_start + los_cutoff
        self.test_start = self.train_end
        self.test_end = self.test_start + params.test_width

        self.train_window = slice(self.train_start,self.train_end)
        self.train_test_window = slice(self.train_start,self.test_end)
        self.test_window = slice(self.test_start,self.test_end)

    

# Specify transition rate manually
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
xs = np.arange(len(df))
manual_transition_rates = np.interp(xs,manual_points[:,0],manual_points[:,1])
plt.plot(manual_transition_rates,label="Manual Transition Rates")
plt.title("Manual Transition Rates")
plt.xticks(xtick_pos[::4],xtick_label[::4])
show_plt()

#%%
# Fitting the LoS curves, as well as the delay and probability
debug_windows = False
debug_distros = False
only_linear = False
less_windows = False

nono = ["beta","invgauss","gamma","weibull","lognorm"] + ["sentinel","block"]

from los_fitter import fit_SEIR
from convolutional_model import calc_its_convolution
from los_fitter import generate_kernel, fit_kernel_distro_to_data_with_previous_results,fit_kernel_to_series
from compartmental_model import calc_its_comp

distro_to_fit = list(distributions.keys())
distro_to_fit += ["SEIR"]
distro_to_fit = [distro for distro in distro_to_fit if distro not in nono]
if debug_distros:
    distro_to_fit = ["linear","SEIR"]
if only_linear:
    distro_to_fit = ["linear"]


fit_results_by_window = []

trans_rates = []
delay = []

if params.fit_admissions:
    if params.smooth_data:
        x_full = df["new_icu_smooth"]
    else:
        x_full = df["new_icu"]
else:
    if params.smooth_data:
        x_full = df["AnzahlFall"]
    else:
        x_full = df["daily"]

y_full = df["icu"].values
x_full = x_full.values

l = list(enumerate(windows))
if less_windows:
    l = l[:3]
elif debug_windows:
    l = [l[10]]
kernels_per_week = {}

for distro in distro_to_fit:
    if distro == "SEIR":
        kernels_per_week[distro] = None
    kernels_per_week[distro] = np.zeros((x_full.shape[0],kernel_width))

first_loop = True
for window_counter, window in l:
    # print("#"*50)
    print(f"Window {window_counter + 1}/{len(windows)}")
    # print("#"*50)
    w = WindowInfo(window)
    if w.test_end >= len(df):
        continue
        
    x_test = x_full[w.train_test_window] #TODO: Rework so that x_test only contains the test window.
    y_test = y_full[w.train_test_window]

    x_train = x_full[w.train_window]
    y_train = y_full[w.train_window]

    init_transition_rate = 0.014472729492187482
    init_delay = 2

    curve_init_params = [init_transition_rate, init_delay]
    curve_fit_boundaries = [(0, 1),(0,10)]
    if params.use_manual_transition_rate:
        transition_rate = manual_transition_rates[windows[i]]
        curve_init_params = [transition_rate, init_delay]
        curve_fit_boundaries = [
            (transition_rate*.9,transition_rate*1.1),
            (init_delay,init_delay)
        ]
    if params.fit_admissions:
        curve_init_params = [1, 0]
        curve_fit_boundaries = [(1, 1),(0,0)]

    
    fit_results = {}
    
    for distro_counter,distro in enumerate(distro_to_fit):
        # print(f"Fitting {distro} - {distro_counter+1}/{len(distro_to_fit)}")
        
        init_values = None    
        if params.reuse_last_parametrization:
            for fr in reversed(fit_results_by_window):
                fr = fr[distro]
                if "params" in fr:
                    init_values = fr['params'][2:]
                    break

            
        if init_values is None:
            if distro in df_init.index:
                init_values = df_init.loc[distro]["params"]
            else:
                init_values = []

        boundaries = [(val,val) for val in init_values]
        try:
            if distro == "SEIR":
                result_dict= fit_SEIR(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    initial_guess_comp=[1/7,1,0],
                    los_cutoff=los_cutoff,
                    )
                y_pred = calc_its_comp(x_full,*result_dict["params"],y_full[0])
            else:      
                
                past_kernels = None          
                if not first_loop:
                    past_kernels = kernels_per_week[distro][w.train_start:w.train_start + los_cutoff]              

                result_dict = fit_kernel_to_series(
                    distro,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    kernel_width,
                    los_cutoff,
                    curve_init_params,
                    curve_fit_boundaries,
                    distro_init_params=init_values,
                    past_kernels = past_kernels,
                    error_fun=params.error_fun,
                    )
                
                if first_loop:
                    kernels_per_week[distro][:] = result_dict["kernel"]
                else:
                    kernels_per_week[distro][w.train_start:] = result_dict["kernel"]
                y_pred = calc_its_convolution(x_full, kernels_per_week[distro], *result_dict["params"][:2],los_cutoff)

            trans_rates.append(result_dict["params"][1])
            delay.append(result_dict["params"][0])

            relative_errors = np.abs(y_pred-y_full)/(y_full+1)
            result_dict["train_relative_error"] = np.mean(relative_errors[w.train_window])
            result_dict["test_relative_error"] =  np.mean(relative_errors[w.test_window])

        except Exception as e:            
            print(f"\tError in {distro}:",e)
            # import traceback
            # traceback.print_exc()
            min_result = types.SimpleNamespace()
            min_result.success = False
            result_dict = {"minimization_result":min_result}

        if result_dict["minimization_result"].success == False:
            print(f"\tFailed to fit {distro}")
        result_dict["success"] = result_dict["minimization_result"].success
        fit_results[distro] = result_dict

    fit_results_by_window.append(fit_results)
    first_loop = False


#%%
# # Run models on a pulse
# debug = True

# path = animation_folder+"/alt_animation/"
# if os.path.exists(path):
#     import shutil
#     shutil.rmtree(path)
# os.makedirs(path)
# ran = range(len(fit_results_by_window))
# if debug:
#     ran = ran[10:12]
# for i in ran:
#     print(f"Print kernels {i+1}/{len(ran)}")
#     fit_results = fit_results_by_window[i]
#     n = 60
#     x_in = np.zeros(n)
#     x_in[0] = 100
#     fig,axs = plt.subplots(3,3,sharex=True,figsize=(15,10))
#     flaxs = axs.flatten()
#     all_kernel_ax = axs[2][0]
#     all_occ_ax = axs[2][1]
#     exp_ax = axs[2][2]
#     for distro,ax in zip(distro_to_fit,flaxs):
#         res = fit_results[distro]
#         if distro == "SEIR":
#             y_pred = calc_its_comp(x_in,*res["params"],0)
#         else:
#             y_pred = calc_its_convolution(x_in, res["kernel"], *res["params"][:2],los_cutoff=0)
#         ax.plot(y_pred[:n],label=f"{distro} - Bed Occupancy")
#         ax.set_title(distro)

#         all_occ_ax.plot(y_pred[:n],label=distro)
#         all_kernel_ax.plot(res["kernel"][:n],label=distro)
#         ax.plot(res["kernel"][:n]*899,color="black",linestyle="--",label=f"{distro} - Kernel (scaled)")
#         ax.set_ylim(-5,101)
#         ax.legend()
#         if distro in ["SEIR","exponential"]:
#             exp_ax.plot(y_pred[:n],label=distro)                
        
#     all_kernel_ax.legend()
#     all_occ_ax.legend()
#     exp_ax.legend()
#     all_kernel_ax.set_title("All Kernels (unscaled)")
#     all_occ_ax.set_title("All Occupancies")
#     exp_ax.set_title("EXP,SEIR Occupancy")
#     plt.suptitle(f"Deconvolution kernels at {window}\n{run_name.replace('_',' ')}",fontsize=16)
#     plt.tight_layout()
#     plt.savefig(path + f"kernels_{windows[i]}")
#     if debug:
#         plt.show()    
#     else:
#         plt.clf()

#%%

# Just fit SEIR-Model
from los_fitter import fit_SEIR
from scipy.optimize import minimize
fig,ax = plt.subplots(1,1,figsize=(10,7),sharex=True)
initial_guess_comp = [1/7,0.02,0]
ax.plot(y_full,label="Real",color="black")
method = "L-BFGS-B"

for window in windows:
    w = WindowInfo(window)
    if w.test_end >= len(df):
        continue
        
    x_test = x_full[w.train_test_window]
    y_test = y_full[w.train_test_window]

    x_train = x_full[w.train_window]
    y_train = y_full[w.train_window]

    result = fit_SEIR(x_train, y_train,x_test,y_test, initial_guess_comp,
                      los_cutoff=los_cutoff)
    y_pred_b = calc_its_comp(x_test, *result["params"],y_test[0])
    xs = np.arange(w.train_start,w.train_end)
    # ax.plot(xs,y_pred[:len(xs)],color=colors[0])
    xs2 = np.arange(w.train_end,w.test_end)
    ax.plot(xs2,y_pred_b[len(xs):],color=colors[1])
# calc_its_comp(inc, discharge_rate, transition_rate, delay,init):
ax.set_xlim(600,1300)
plt.title("SEIR-Models")
plt.show()


#%% Generate Video 2
##animation

debug = True
if True:
    if not debug:
        path = animation_folder
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path)

    window_counter = 1
    stuff = list(zip(windows,fit_results_by_window))
    if debug:
        stuff = [stuff[min(10,len(stuff)-1)]]
    for window, fit_results in stuff:
        print(f"Animation Window {window_counter}/{len(windows)}")
        window_counter+=1
        fig = plt.figure(figsize=(17, 10),dpi=150)
        gs = gridspec.GridSpec(2, 4,height_ratios=[2,1])
        ax1 = fig.add_subplot(gs[0, :4])
        ax11 = ax1.twinx()
        ax2 = fig.add_subplot(gs[1,:2])
        ax3 = fig.add_subplot(gs[1, 2])
        ax32 = fig.add_subplot(gs[1, 3])

        w = WindowInfo(window)

        line1, = ax1.plot(y_full, color="black",label="ICU Bedload")
        ax12 = ax1.twinx()

        mutant_lines = ax12.plot(df_mutant.values)

        span1 = ax1.axvspan(w.train_start, w.train_los_cutoff, color="magenta", alpha=0.1,label=f"Los convolution edge window")
        span2 = ax1.axvspan(w.train_los_cutoff, w.train_end, color="red", alpha=0.2,label=f"Training = {params.train_width-los_cutoff} days")
        span3 = ax1.axvspan(w.test_start, w.test_end, color="blue", alpha=0.05,label=f"Test")
        ax1.axvline(w.train_end,color="black",linestyle="-",linewidth=1)

        label = "COVID Incidence (scaled)"
        if params.fit_admissions:
            label = "New ICU Admissions (scaled)"
        line2, = ax11.plot(x_full,linestyle="--",label=label)
        # use scientific noatation for y axis
        ax11.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        ma = np.nanmax(x_full)
        ax11.set_ylim(-ma/7.5,ma*4)

        plot_lines  = []
        replace_names = {"block":"Constant Discharge","sentinel":"Baseline: Sentinel"}
        for distro in distro_to_fit:
            result = fit_results[distro]
            name = replace_names.get(distro,distro.capitalize())
            if "minimization_result" in result and result["minimization_result"].success == False:
                ax2.plot([],[],label=name)
                l, = ax1.plot([],[],label=name)
                plot_lines.append(l)
                i+=1
                continue
            ax2.plot(result['kernel'], label=name)
            c = ax2.get_lines()[-1].get_color()
            y = result['curve'][los_cutoff:]
            
            s = np.arange(len(y))+los_cutoff+w.train_start
            l, = ax1.plot(s,y, label=f"{distro.capitalize()}")
            plot_lines.append(l)
            
        ax2.plot(real_los, color="black",label="Sentinel LoS Charité")
        
        legend1 = ax1.legend(handles = plot_lines,loc="upper left")
        legend2 = ax1.legend(handles = [line1, line2, span1, span2, span3],loc="upper right")
        legend3 = ax1.legend(mutant_lines,df_mutant.columns,loc="lower right")
        ax1.add_artist(legend1)
        ax1.add_artist(legend2)
        ax1.add_artist(legend3)

        ax1.set_title(f"ICU Occupancy")
        ax1.set_xticks(xtick_pos)
        ax1.set_xticklabels(xtick_label)
        if params.fit_admissions:
            ax1.set_xlim(new_icu_day-30,1300)
        else:
            ax1.set_xlim(70,1300)
        ax1.set_ylim(-200,6000)
        ax1.set_ylabel("Occupied Beds")
        ax11.set_ylabel("(Incidence)")
        ax2.legend(loc = "upper right",fancybox=True,ncol=2)
        ax2.set_ylim(0,0.1)
        ax2.set_xlim(-2,80)
        ax2.set_ylabel("Discharge Probability")
        ax2.set_xlabel("Days after admission")
        ax2.set_title(f"Estimated LoS Kernels")


        train_errors = [fit_results[distro]['train_relative_error'] for distro in distro_to_fit]
        test_errors = [fit_results[distro]['test_relative_error'] for distro in distro_to_fit]

        nan_val = -1e5
        train_errors = [nan_val if error == np.inf else error for error in train_errors]
        test_errors =  [nan_val if error == np.inf else error for error in test_errors]
        
        for i, distro in enumerate(distro_to_fit):
            patch, = ax3.bar([distro],train_errors[i],label="Train",color=colors[i])
            c = patch.get_facecolor()
            ax32.bar([distro],test_errors[i],label="Test",color=c)


        lim = .4
        ax3.set_ylim(0,lim)
        ax3.set_title("Relative Train Error")
        ax3.set_xticks(distro_to_fit)        
        ax3.set_xticklabels(distro_to_fit,rotation=75)
        ax3.set_ylabel("Relative Error")

        ax32.set_ylim(0,lim)
        ax32.set_title("Relative Test Error")
        ax32.set_xticks(distro_to_fit)
        ax32.set_xticklabels(distro_to_fit,rotation=75)
        ax32.set_ylabel("Relative Error")

        plt.suptitle(f"Deconvolution Training Process\n{run_name.replace('_',' ')}",fontsize=16)
        plt.tight_layout()
        if debug:
            show_plt()
        else:
            plt.savefig(animation_folder + f"fit_{window:04d}.png")
            plt.close()
        plt.clf()

#%%
#Save results
import pickle
with open(results_folder + "fit_results.pkl","wb") as f:
    pickle.dump(fit_results_by_window,f)
#%%
# Plot number of failed fits and successfull fits
n_success = []
for distro in distro_to_fit:
    n = sum([1 for fit_results in fit_results_by_window if fit_results[distro]["minimization_result"].success == True])
    n_success.append(n)
plt.figure(figsize=(10,5),dpi=150)
plt.bar(distro_to_fit,n_success)
plt.title("Number of successful fits")
plt.axhline(len(fit_results_by_window),color="red",linestyle="--",label="Total")
plt.xticks(rotation=45)
plt.savefig(figures_folder + "successful_fits.png")

show_plt()

#%%
# Calculate success rate 
train_errors_by_distro = [[fit[distro]['train_relative_error'] for fit in fit_results_by_window] for distro in distro_to_fit]
test_errors_by_distro = [[fit[distro]['test_relative_error'] for fit in fit_results_by_window] for distro in distro_to_fit]
success_by_distro = [[fit[distro]["minimization_result"].success for fit in fit_results_by_window] for distro in distro_to_fit]
failure_by_distro = [[0 if success else 1 for success in successes] for successes in success_by_distro]

# Convert losses to DataFrame
df_train = pd.DataFrame(np.array(train_errors_by_distro).T, columns=distro_to_fit)
df_test = pd.DataFrame(np.array(test_errors_by_distro).T, columns=distro_to_fit)

df_failures = pd.DataFrame(np.array(failure_by_distro).T, columns=distro_to_fit)
# Compute mean finite loss and failure rate for each model
summary = pd.DataFrame(index=distro_to_fit)
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

#%%
# Find outliers
for distro in distro_to_fit:
    Q1 = df_test[distro].quantile(0.25)
    Q3 = df_test[distro].quantile(0.75)
    IQR = Q3 - Q1
    # filter out outliers
    mask = (df_test[distro] < (Q1 - 1.5 * IQR)) | (df_test[distro] > (Q3 + 1.5 * IQR))
    summary.at[distro,col] = df_test[distro][~mask].mean()

#%%
# save test errors and failure rates in csv
df_test.to_csv(results_folder + "test_errors.csv")
df_failures.to_csv(results_folder + f"failure_rates.csv")
summary.to_csv(results_folder + "summary.csv")

#%%

# Visualization
def viz(col2, col1,ylim=None,save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6),dpi=150)

    for i, distro in enumerate(distro_to_fit):
        if distro in ["sentinel","block"]:
            continue
        val1 = summary[col1][distro]
        val2 = summary[col2][distro]
        ax.scatter(val1, val2, s=100, label=distro, color=colors[i])
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
#remove inf from both
train_errors_by_distro = [[error for error in errors if error != np.inf] for errors in train_errors_by_distro]
test_errors_by_distro = [[error for error in errors if error != np.inf] for errors in test_errors_by_distro]
#%%

plt.figure(figsize=(10,5),dpi=150)
plt.boxplot(train_errors_by_distro)
distro_and_n = [f"{distro.capitalize()} n={succs}" for distro,succs in zip(distro_to_fit,n_success)]
plt.xticks(np.arange(len(distro_to_fit))+1,distro_and_n,rotation=45)
plt.title("Train Error")
plt.ylabel("Relative Train Error")
plt.tight_layout()
plt.savefig(figures_folder + "train_error_boxplot.png")
show_plt()

plt.figure(figsize=(10,5),dpi=150)
plt.boxplot(test_errors_by_distro)
plt.xticks(np.arange(len(distro_to_fit))+1,distro_and_n,rotation=45)
plt.title(f"Test Error\n{run_name}")
plt.ylabel(f"Relative Error")
plt.tight_layout()
plt.savefig(figures_folder + "test_error_boxplot.png")
show_plt()
#%%
fig = plt.figure(figsize=(10,5))
sns.stripplot(data=train_errors_by_distro, jitter=0.2)
plt.xticks(np.arange(len(distro_to_fit)),distro_to_fit,rotation=45)
plt.title(f"Train Error\n{run_name}")
plt.savefig(figures_folder + "train_error_stripplot.png")
show_plt()



#%%
for distro in distro_to_fit:
    
    fig,(ax,ax4,ax2)= plt.subplots(3,1,figsize=(10,5),sharex=True,dpi=150)
    ax.plot(y_full, color="black",label="Real" ,alpha=.8,linestyle="--")
    for i,fit_results in enumerate(fit_results_by_window):
        window = windows[i]
        if fit_results[distro]["minimization_result"].success == False:
            start, end = window-params.train_width, window
            ax.axvspan(start,end, color="red", alpha=0.1)
            continue
        result = fit_results[distro]
        y = result['curve']
        w = WindowInfo(window)
        
        _y = y[los_cutoff:params.train_width]
        l1, = ax.plot(
            np.arange(w.train_los_cutoff,w.train_end)[:len(_y)],
            _y,
            color=colors[0],
        )
        _y = y[params.train_width:params.train_width+params.test_width]
        l2, = ax.plot(
            np.arange(w.train_end,w.test_end)[:len(_y)],
            _y,
            color=colors[1],
        )

    
    ax.plot([],[],color=colors[0],label = f"{distro.capitalize()} Train")
    ax.plot([],[],color=colors[1],label = f"{distro.capitalize()} Prediction")
    ax.axvspan(0,0, color="red", alpha=0.1,label="Failed Training Windows")
    ax.axvspan(sentinel_start_day,sentinel_end_day, color="green", alpha=0.1,label="Sentinel Window")
    ax.legend(loc="upper left")
    ax.set_ylim(-100,6000)    
    ax.set_xticks(xtick_pos[::2])
    ax.set_xticklabels(xtick_label[::2])
    if params.fit_admissions:
        ax.set_xlim(new_icu_day-80,1300)
    else:
        ax.set_xlim(50,1300)
    ax.grid()
    
    trans_probs = np.zeros(len(fit_results_by_window))*np.nan
    trans_delay = np.zeros(len(fit_results_by_window))*np.nan
    _train_errs = np.zeros(len(fit_results_by_window))*np.nan
    _test_errs = np.zeros(len(fit_results_by_window))*np.nan
    for i,fit_results in enumerate(fit_results_by_window):
        if distro =="SEIR":
            _train_errs[i] = fit_results[distro]["train_error"][0]
            _test_errs[i] = fit_results[distro]["test_error"][0]
        else:
            ####################################################################################
            #TODo: HACK
            ####################################################################################
            _train_errs[i] = fit_results[distro]["train_error"]
            _test_errs[i] = fit_results[distro]["test_error"]

        if fit_results[distro]["minimization_result"].success == False:
            continue
        trans_probs[i] = fit_results[distro]["params"][0]
        trans_delay[i] = fit_results[distro]["params"][1]

    # plot trans rates in ax2
    ax2.bar(windows,trans_probs, width=15 ,label="Transition Probability")
    ax2.grid()
    ax2.set_ylim(-.01,0.1)
    ax2.set_title("Transition Probability")

    ax4.plot(windows,_train_errs,label = "Train Error")
    ax4.plot(windows,_test_errs,label = "Test Error")
    # mark nan and inf values
    for i in range(len(windows)):
        if fit_results_by_window[i][distro]["minimization_result"].success == False:
            ax4.axvline(windows[i],color="red",alpha=.5)
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
ax.plot([],[],color=colors[0],label = f"{distro.capitalize()} Train")
ax.plot([],[],color=colors[1],label = f"{distro.capitalize()} Prediction")
ax.axvspan(0,0, color="red", alpha=0.1,label="Failed Training Windows")
for distro in distro_to_fit:

    for i,fit_results in enumerate(fit_results_by_window):
        window = windows[i]
        if fit_results[distro]["minimization_result"].success == False:
            start, end = window-params.train_width, window
            ax.axvspan(start,end, color="red", alpha=0.01)
            continue

        result = fit_results[distro]
        y = result['curve']
        w = WindowInfo(window)

        _y = y[los_cutoff:params.train_width]
        l1, = ax.plot(
            np.arange(w.train_los_cutoff,w.train_end)[:len(_y)],
            _y,
            color=colors[0],
        )
        _y = y[params.train_width:params.train_width+params.test_width]
        l2, = ax.plot(
            np.arange(w.train_end,w.test_end)[:len(_y)],
            _y,
            color=colors[1],
        )




    trans_probs = np.zeros(len(fit_results_by_window))*np.nan
    trans_delay = np.zeros(len(fit_results_by_window))*np.nan
    _train_errs = np.zeros(len(fit_results_by_window))*np.nan
    _test_errs = np.zeros(len(fit_results_by_window))*np.nan
    for i,fit_results in enumerate(fit_results_by_window):
        if distro =="SEIR":
            _train_errs[i] = fit_results[distro]["train_error"][0]
            _test_errs[i] = fit_results[distro]["test_error"][0]
        else:
            ####################################################################################
            #TODo: HACK
            ####################################################################################
            _train_errs[i] = fit_results[distro]["train_error"]
            _test_errs[i] = fit_results[distro]["test_error"]
            
        if fit_results[distro]["minimization_result"].success == False:
            continue
        trans_probs[i] = fit_results[distro]["params"][0]
        trans_delay[i] = fit_results[distro]["params"][1]




    ax4.plot(windows,_train_errs,color=colors[0])
    ax4.plot(windows,_test_errs, color=colors[1])

ax.set_ylim(-100,6000)
if params.fit_admissions:
    ax.set_xlim(new_icu_day-80,1300)
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
if params.fit_admissions:
    ax.set_xlim(new_icu_day-30,1300)
ax.set_title(f"All Predictions\n{run_name}")
plt.savefig(figures_folder + "prediction_error_all_distros.png")
show_plt()

#%%
fig,ax= plt.subplots(1,1,figsize=(15,7.5),sharex=True)
ax.plot(y_full, color="black",label="Real" ,alpha=.8,linestyle="--")

for distro in distro_to_fit:
    for i,fit_results in enumerate(fit_results_by_window):
        if fit_results[distro]["minimization_result"].success == False:
            continue
        window = windows[i]
        w = WindowInfo(window)
        result = fit_results[distro]
        y = result['curve']

        _y = y[los_cutoff:params.train_width]
        l1, = ax.plot(
            np.arange(w.train_los_cutoff,w.train_end)[:len(_y)],
            _y,
            color=colors[0],
        )
        _y = y[params.train_width:params.train_width+params.test_width]
        l2, = ax.plot(
            np.arange(w.train_end,w.test_end)[:len(_y)],
            _y,
            color=colors[1],
        )

ax.set_ylim(-100,6000)
if params.fit_admissions:
    ax.set_xlim(new_icu_day-30,1300)
ax.grid()

ax.plot([],[],color=colors[0],label = f"All Distros Train")
ax.plot([],[],color=colors[1],label = f"All Distros Prediction")
ax.legend()
ax.set_title(f"All Models\n{run_name}")
plt.savefig(figures_folder + "prediction_error_all_fits.png")
show_plt()


# %%
# plot distributions

for distro in distro_to_fit:
    fig, ax = plt.subplots(figsize=(10,5))
    # plot real kernel
    ax.plot(real_los,color='black',label="Real")

    for result in fit_results_by_window:
        if result[distro]["minimization_result"].success == False:
            continue
        y = result[distro]["kernel"]
        ax.plot(y,alpha=0.3,color=colors[0])

    plt.grid()
    plt.legend()
    plt.title(f"{distro.capitalize()} Kernel\n{run_name}")
    plt.ylim(-0.005,0.3)
    plt.tight_layout()
    plt.savefig(figures_folder + f"all_kernels_{distro}.png")
    show_plt()
#%%



if "sentinel" in distro_to_fit:
    from scipy.optimize import minimize

    # Fit transition rates manually
    tr = np.array([fit_results["sentinel"]["params"][0] for fit_results in fit_results_by_window])
    tr = np.insert(tr,0,tr[0])


    # Fit transition rates manually
    tr = np.array([fit_results["sentinel"]["params"][0] for fit_results in fit_results_by_window])
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
    xs = np.arange(len(df))
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
    xs = np.arange(len(df))
    ys = np.interp(xs,manual_points[:,0],manual_points[:,1])

    print(obj_fun(result.x))
    print(result.x.reshape(-1,2))
# %%
transition_rates = np.array([[fit_results[distro]["params"][0] for fit_results in fit_results_by_window] for distro in distro_to_fit])

# transition rates in a subplot together with the icu
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,10),sharex=True)
# ax1.plot(df["icu"].values,label="ICU")
y = df["icu"].values/df["AnzahlFall"].values
ax1.plot(y,label="icu incidence ratio")
ax1.set_title("ICU Occupancy")
ax1.legend()
dddd =['lognorm',
'weibull',
'gaussian',
'exponential',
'gamma',
'beta',
'cauchy',
't',
'invgauss',
'block'
]
nono=[
    "weibull",
    "block"
]
for i,distro in enumerate(distro_to_fit):
    if distro in nono:
        continue
    ax2.plot(windows,transition_rates[i],label=distro)
ax2.axhline(0,color="black",linestyle="--")
ax2.legend()
ax2.set_title("Transition Rates")
# ax2.set_ylim(-.03,0.2)
plt.savefig(figures_folder + "transition_rates_and_icu.png")
show_plt()
#%%
# Generate ensemble of predictions
all_predictions_combined = []
for i, fit_results in enumerate(fit_results_by_window):
    window = windows[i]
    pred = np.zeros((len(df),len(distro_to_fit)))*np.nan
    for window_counter,distro in enumerate(distro_to_fit):
        if fit_results[distro]["minimization_result"].success == False:
            continue
        result = fit_results[distro]
        y = result['curve']
        w = WindowInfo(window)
        _y = y[params.train_width:params.train_width+params.test_width]
        pred[w.train_end:w.test_end,window_counter] = _y
    all_predictions_combined.append(pred)
all_predictions_combined = np.array(all_predictions_combined)
all_predictions_combined.shape # in shape (n_windows, n_days, n_distros)

#%%
distros = []
for i, distro in enumerate(distro_to_fit):
    if distro in ["cauchy","exponential","t","gaussian","sentinel"]:
        distros.append((i,distro))
fig,(ax,ax2) = plt.subplots(2,1,figsize=(10,5),sharex=True,dpi=150)
# plot icu and incidence
ax.plot(y_full, color="black",label="ICU Bedload")
for i, window in enumerate(windows):
    w = WindowInfo(window)
    x = np.arange(w.train_end,w.test_end)
    for j, distro in distros:
        y = all_predictions_combined[i,w.train_end:w.test_end,j]
        ax.plot(x,y,color=colors[j],alpha=1)
ax.set_ylim(-100,6000)
ax.set_title("All Predictions")
ax.legend()

transition_rates = np.array([[fit_results[distro]["params"][0] for fit_results in fit_results_by_window] for distro in distro_to_fit])
lines =[]
for i,distro in distros:
    l, = ax2.plot(windows, transition_rates[i],label=distro)
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
for i, distro in enumerate(distro_to_fit):
    train_errors = [fit_results[distro]["train_error"] for fit_results in fit_results_by_window]
    test_errors = [fit_results[distro]["test_error"] for fit_results in fit_results_by_window]
    ax.plot(windows,train_errors,label=f"{distro.capitalize()} Train")
    ax.plot(windows,test_errors,label=f"{distro.capitalize()} Test")
ax.legend()
ax.set_title("All Errors")
ax.set_xticks(xtick_pos[::2])
ax.set_xticklabels(xtick_label[::2])
plt.grid()

#%%
os.system(f'msg * "Finished Run - {run_name}"')

# %%
# calculate 90% containment for all kernels
plt.figure(figsize=(10,5),dpi=150)
cutoffs = []
for distro in distro_to_fit:
    if distro in ["block","sentinel"]:
        continue
    for result in fit_results_by_window:
        if result[distro]["minimization_result"].success == False:
            continue
        y = result[distro]["kernel"]
        # find the point, where 90% of the probability mass is contained    
        cumsum = np.cumsum(y)
        idx = np.argmax(cumsum > 0.99)
        cutoffs.append(idx)
        plt.plot(y[:idx],alpha=0.1)
plt.title("90% Containment")
plt.grid()
plt.show()
plt.plot(sorted(cutoffs))
plt.axvline(len(cutoffs)//2,color="red",linestyle="--")
plt.axvline(len(cutoffs)//4*3,color="red",linestyle="--")
plt.plot()
#%%