#%%
# reload imports
import os
import sys
import numpy as np
import pandas as pd
import types
import seaborn as sns
import shutil

import time
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.append("../02_fit_los_distributions/")
from dataprep import load_los, load_incidences, load_icu_occupancy, load_mutant_distribution
from compartmental_model import calc_its_comp
from los_fitter import distributions, calc_its_convolution, fit_SEIR
#%%


def get_graph_colors():
    # take matplotlib standart color wheel
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # add extra color palette
    colors += ["#FFA07A","#20B2AA","#FF6347","#808000","#FF00FF","#FFD700","#00FF00","#00FFFF","#0000FF","#8A2BE2"]
    return colors


def load_inc_beds(start_day, end_day):
    df_inc, raw = load_incidences(start_day, end_day)
    # wenn die ersten beiden Yeilen der 1.1.2020 sind entferne eine
    if df_inc.index[0] == pd.Timestamp("2020-01-01") and df_inc.index[1] == pd.Timestamp("2020-01-01"):
        df_inc = df_inc.iloc[1:]

    df_icu = load_icu_occupancy(start_day, end_day)
    df = df_inc.join(df_icu,how="inner")
    df["new_icu_smooth"] = df["new_icu"].rolling(7).mean()
    return df


def generate_xticks(df):
    xtick_pos = []
    xtick_label = []
    for i in range(0,len(df)):
        if df.index[i].day == 1: # first day of month
            xtick_pos.append(i)
            label = df.index[i].strftime("%b")
            if i == 0 or df.index[i].month == 1:
                label += f"\n{df.index[i].year}"
            xtick_label.append(label)
    return xtick_pos,xtick_label


def load_init_parameters(file):
    df_init = pd.read_csv(file,index_col=0)
    df_init = df_init.set_index("distro")
    # interpret params as array float of format [f1 f2 f3 ...]
    df_init["params"] = df_init["params"].apply(lambda x: [float(i) for i in x[1:-1].split()])
    return df_init


def load_all_data(los_file, init_params_file, mutants_file, start_day, end_day):
    df_occupancy = load_inc_beds(start_day, end_day)
    real_los, _ = load_los(file=los_file)

    df_init = load_init_parameters(init_params_file)
    df_mutant = load_mutant_distribution(mutants_file)
    df_mutant = df_mutant.reindex(df_occupancy.index,method="nearest")

    xtick_pos, xtick_label = generate_xticks(df_occupancy)
    new_icu_date = df_occupancy.index[df_occupancy["new_icu"]>0][0]
    return df_occupancy,real_los,df_init,df_mutant,xtick_pos,xtick_label,new_icu_date


def generate_run_name(params):
    timestamp = time.strftime("%y%m%d_%H%M")
    run_name = f"{timestamp}_dev"

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
    return run_name



# Specify transition rate manually
def get_manual_transition_rates(df_occupancy):
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
    manual_transition_rates = np.interp(xs,manual_points[:,0],manual_points[:,1])
    return manual_transition_rates

def create_result_folders(run_name):
    results_folder = f"./results/{run_name}/"
    figures_folder = results_folder + "/figures/"
    animation_folder = results_folder + "/animation/"

    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)    
    os.makedirs(results_folder)
    os.makedirs(figures_folder)
    os.makedirs(animation_folder)
    return results_folder,figures_folder,animation_folder



def run_pulse_model(run_name, animation_folder, windows, distro_to_fit, fit_results_by_window, window, debug):
    path = animation_folder+"/alt_animation/"
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    os.makedirs(path)
    ran = range(len(fit_results_by_window))
    if debug:
        ran = ran[10:12]
    for i in ran:
        print(f"Print kernels {i+1}/{len(ran)}")
        fit_results = fit_results_by_window[i]
        n = 60
        x_in = np.zeros(n)
        x_in[0] = 100
        fig,axs = plt.subplots(3,3,sharex=True,figsize=(15,10))
        flaxs = axs.flatten()
        all_kernel_ax = axs[2][0]
        all_occ_ax = axs[2][1]
        exp_ax = axs[2][2]
        for distro,ax in zip(distro_to_fit,flaxs):
            res = fit_results[distro]
            if distro == "SEIR":
                y_pred = calc_its_comp(x_in,*res["params"],0)
            else:
                y_pred = calc_its_convolution(x_in, res["kernel"], *res["params"][:2],los_cutoff=0)
            ax.plot(y_pred[:n],label=f"{distro} - Bed Occupancy")
            ax.set_title(distro)

            all_occ_ax.plot(y_pred[:n],label=distro)
            all_kernel_ax.plot(res["kernel"][:n],label=distro)
            ax.plot(res["kernel"][:n]*899,color="black",linestyle="--",label=f"{distro} - Kernel (scaled)")
            ax.set_ylim(-5,101)
            ax.legend()
            if distro in ["SEIR","exponential"]:
                exp_ax.plot(y_pred[:n],label=distro)                
        
        all_kernel_ax.legend()
        all_occ_ax.legend()
        exp_ax.legend()
        all_kernel_ax.set_title("All Kernels (unscaled)")
        all_occ_ax.set_title("All Occupancies")
        exp_ax.set_title("EXP,SEIR Occupancy")
        plt.suptitle(f"Deconvolution kernels at {window}\n{run_name.replace('_',' ')}",fontsize=16)
        plt.tight_layout()
        plt.savefig(path + f"kernels_{windows[i]}")
        if debug:
            plt.show()    
        else:
            plt.clf()
