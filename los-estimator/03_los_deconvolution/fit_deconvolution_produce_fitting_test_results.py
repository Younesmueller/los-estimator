import os
import sys
import numpy as np
import pandas as pd
import types

from tqdm import tqdm
from compartmental_model import calc_its_comp
from los_fitter import calc_its_convolution, fit_SEIR
from fit_deconvolution_functions import *
from core import *
from high_vizzz import *
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from los_fitter import fit_SEIR, fit_kernel_to_series, SingleFitResult
from convolutional_model import calc_its_convolution
from compartmental_model import calc_its_comp

def produce_fitting_test_result(df_init, params, series_data,distributions,MultiSeriesFitResults,
                                DEBUG_WINDOWS = False,
                                DEBUG_DISTROS = False,
                                ONLY_LINEAR = False,
                                LESS_WINDOWS = False):
    def select_init_params(all_fit_results, distro, params, window_id):
        if distro == "compartmental":
            a = 3
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


# --- Configuration flags (could come from argparse or a config object) ---
    

# Distributions we explicitly skip
    EXCLUDE_DISTROS = {"beta", "invgauss", "gamma", "weibull", "lognorm", "sentinel", "block"}

# --- Build list of distros to fit ---
    all_distros = [d for d in distributions if d not in EXCLUDE_DISTROS]
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
    d: (None if d == "compartmental" else np.zeros((series_data.n_days, params.kernel_width)))
    for d in distro_to_fit
}

    all_fit_results = MultiSeriesFitResults(distro_to_fit)
    trans_rates = []
    delays = []


# --- Main loop ---

    for distro in distro_to_fit:
        print(f"Distro: {distro}")
        failed_windows = []
        first_window = True
    # SEIR always uses its own fitter
        for window_id, window_info, train_data, test_data in tqdm(window_data):
            w = window_info

        # Build curve_init and boundary tuples if needed
            curve_init, curve_bounds = None, None
            if params.fit_admissions:
                curve_init = [1, 0]
                curve_bounds = [(1, 1), (0, 0)]

        # per‐distro initialization
            init_vals = select_init_params(all_fit_results, distro, params, window_id)
            distro_bounds = [(v, v) for v in init_vals]

            try:
                if distro == "compartmental":
                    result_dict, result_obj = fit_SEIR(
                    *train_data,*test_data,
                    initial_guess_comp=[1/7, 1, 0],
                    los_cutoff=params.los_cutoff,
                )
                    y_pred = calc_its_comp(series_data.x_full, *result_obj.params, series_data.y_full[0])
                else:
                    past_kernels = None
                    if not first_window and params.variable_kernels:
                        past_kernels = kernels_per_week[distro][w.train_start : w.train_start + params.los_cutoff]
                    result_dict, result_obj = fit_kernel_to_series(
                    distro,
                    *train_data,*test_data,
                    params.kernel_width, params.los_cutoff,
                    curve_init, curve_bounds,
                    distro_init_params=init_vals,
                    past_kernels=past_kernels,
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
                    series_data.x_full, kernel_full, *result_obj.params[:2], params.los_cutoff
                )

            # record transition & delay
                trans_rates.append(result_obj.params[1])
                delays.append(result_obj.params[0])

            # compute errors
                rel_err = np.abs(y_pred - series_data.y_full) / (series_data.y_full + 1)
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
    return window_data,all_fit_results
