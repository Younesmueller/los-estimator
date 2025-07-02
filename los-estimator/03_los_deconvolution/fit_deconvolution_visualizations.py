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
print("Let's Go!")



def visualize_fit_deconvolution(
        all_fit_results,
        series_data,
        params,
        vd,
        df_mutant_selection = None,
        DEBUG_ANIMATION = True,
        DEBUG_HIDE_FAILED = True,
        ):
    SHOW_MUTANTS = df_mutant_selection is not None

    if not DEBUG_ANIMATION:
        path = vd.animation_folder
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path)

    window_counter = 1
    to_enumerate = list(enumerate(series_data.window_infos))
    if DEBUG_ANIMATION:
        xx = 2
        to_enumerate = [to_enumerate[min(xx,len(to_enumerate)-1)]]
    for window_id, window_info in to_enumerate:
        print(f"Animation Window {window_counter}/{len(series_data)}")
        window_counter+=1


        if SHOW_MUTANTS:
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
        line_bedload, = ax_main.plot(series_data.y_full, color="black",label="ICU Bedload")
        zero = np.zeros_like(series_data.y_full)

        span_los_cutoff = ax_main.axvspan(w.train_start, w.train_los_cutoff, color="magenta", alpha=0.1,label=f"Train Window (Convolution Edge) = {params.train_width} days")
        span_train = ax_main.axvspan(w.train_los_cutoff, w.train_end, color="red", alpha=0.2,label=f"Training = {params.train_width-params.los_cutoff} days")
        span_test = ax_main.axvspan(w.test_start, w.test_end, color="blue", alpha=0.05,label=f"Test Window = {params.test_width} days")
        ax_main.axvline(w.train_end,color="black",linestyle="-",linewidth=1)

        label = "COVID Incidence (Scaled)"
        if params.fit_admissions:
            label = "New ICU Admissions (Scaled)"


        line_inc, = ax_inc.plot(series_data.x_full,linestyle="--",label=label)
        ax_inc.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        ma = np.nanmax(series_data.x_full)
        ax_inc.set_ylim(-ma/7.5,ma*4)

        plot_lines  = []

        for distro,result_series in all_fit_results.items():
            result_obj = result_series.fit_results[window_id]
            name = vd.replace_names.get(distro,distro.capitalize())
            if DEBUG_HIDE_FAILED and not result_obj.success:
                continue
            c = vd.distro_colors[distro]
            ax_kernel.plot(result_obj.kernel, label=name, color=c)
            y = result_obj.curve[params.los_cutoff:]
            s = np.arange(len(y))+params.los_cutoff+w.train_start
            l, = ax_main.plot(s,y, label=f"{distro.capitalize()}", color=c)
            plot_lines.append(l)

        ax_kernel.plot(vd.real_los, color="black",label="Sentinel LoS Charité")

        for i, (distro, fit_result) in enumerate(all_fit_results.items()):
            c = vd.distro_colors[distro]
            train_err = fit_result.train_relative_errors[window_id]
            test_err = fit_result.test_relative_errors[window_id]

            if DEBUG_HIDE_FAILED and not fit_result[window_id].success:
                ax_err_train.bar(i,1e100,color="lightgrey",hatch="/")
                ax_err_test.bar(i, 1e100,color="lightgrey",hatch="/")
                ax_err_train.bar(i,train_err,color="black")
                ax_err_test.bar(i,test_err,color="black")
                continue
            ax_err_train.bar(i,train_err,label="Train",color=c)
            ax_err_test.bar(i,test_err,label="Test",color=c)

        legend1 = ax_main.legend(handles=vd.distro_patches, loc="upper left", fancybox=True,ncol=2)
        legend2 = ax_main.legend(handles = [line_bedload, line_inc, span_los_cutoff, span_train, span_test],loc="upper right")

        ax_main.add_artist(legend1)
        ax_main.add_artist(legend2)

        ax_main.set_title(f"ICU Occupancy")
        ax_main.set_xticks(vd.xtick_pos)
        ax_main.set_xticklabels(vd.xtick_label)
        ax_main.set_xlim(*vd.xlims)
        ax_main.set_ylim(-200,6000)
        ax_main.set_ylabel("Occupied Beds")

        if SHOW_MUTANTS:
            mutant_lines = []
            for col in df_mutant_selection.columns:
                line, = ax_mutant.plot(df_mutant_selection[col].values)
                mutant_lines.append(line)
                ax_mutant.fill_between(range(len(series_data.y_full)), df_mutant_selection[col].values, 0, alpha=0.3)

            ax_mutant.legend(mutant_lines,df_mutant_selection.columns,loc="upper right")
            ax_mutant.set_xticks([])
            ax_mutant.set_xticklabels([])
            ax_mutant.set_xticks(vd.xtick_pos)
            tmp_xtick = [label.split("\n")[1:] for label in vd.xtick_label]
            tmp_xtick = [label[0] if label else ""  for label in tmp_xtick]
            ax_mutant.set_xticklabels(tmp_xtick)
            ax_mutant.set_xlim(*vd.xlims)
            ax_mutant.set_title("Variants of Concern")
            ax_mutant.set_ylabel("Variant Share (%)")




        ax_inc.set_ylabel("(Incidence)")
        if params.fit_admissions:
            ax_inc.set_ylabel("New ICU Admissions (scaled)")
        ax_kernel.legend(handles=vd.distro_patches, loc="upper right", fancybox=True, ncol=2, )
        ax_kernel.set_ylim(0,0.1)
        ax_kernel.set_xlim(-2,80)
        ax_kernel.set_ylabel("Discharge Probability")
        ax_kernel.set_xlabel("Days after admission")
        ax_kernel.set_title(f"Estimated LoS Kernels")


        lim = .4
        ax_err_train.set_ylim(0,lim)
        ax_err_train.set_title("Relative Train Error")
        ax_err_train.set_xticks(range(len(all_fit_results)))
        ax_err_train.set_xticklabels(vd.short_distro_names,rotation=75)
        ax_err_train.set_ylabel("Relative Error")

        ax_err_test.set_ylim(0,lim)
        ax_err_test.set_title("Relative Test Error")
        ax_err_test.set_xticks(range(len(all_fit_results)))
        ax_err_test.set_xticklabels(vd.short_distro_names,rotation=75)
        ax_err_test.set_ylabel("Relative Error")

        plt.suptitle(f"Deconvolution Training Process\n{params.run_name.replace('_',' ')}",fontsize=16)
        plt.tight_layout()
        if DEBUG_ANIMATION:
            vd.show_plt()
        else:
            plt.savefig(vd.animation_folder + f"fit_{w.train_start:04d}.png")
            plt.close()
        plt.clf()
