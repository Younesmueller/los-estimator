import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from los_estimator.fitting.los_fitter import fit_compartmental, calc_its_comp, fit_convolution, calc_its_convolution, SingleFitResult
from .fit_results import SingleFitResult, SeriesFitResult, MultiSeriesFitResults



class MultiSeriesFitter:
    all_fit_results: MultiSeriesFitResults
    
    def __init__(self, series_data, model_config, distributions: list[str], init_parameters):
        self.series_data = series_data
        self.model_config = model_config        
        self._distributions: list[str] = distributions
        self.distributions: list[str] = None
        self.exclude_distros: set[str] = set()
        self.all_fit_results: MultiSeriesFitResults = MultiSeriesFitResults()
        self.init_parameters = defaultdict(list, init_parameters)
        self.DEBUG_MODE(False,False,False,False)

    def DEBUG_MODE(self,
        ONE_WINDOW:bool = False,
        LESS_WINDOWS:bool = True,
        LESS_DISTROS:bool = False,
        ONLY_LINEAR:bool = False,
        
        ):
        self.DEBUG = {
            "ONE_WINDOW": ONE_WINDOW,
            "LESS_WINDOWS": LESS_WINDOWS,
            "LESS_DISTROS": LESS_DISTROS,
            "ONLY_LINEAR": ONLY_LINEAR,
        }
        self.distributions = self._get_debug_distro(self._distributions)
        self.window_data = self._get_debug_window_data(self.series_data)


    def _get_debug_distro(self, distributions):
        if self.DEBUG["LESS_DISTROS"]:
            return ["linear", "compartmental"]
        if self.DEBUG["ONLY_LINEAR"]:
            return ["linear"]
        return distributions
    
    def _get_debug_window_data(self, series_data):
        window_data = list(series_data)
        if self.DEBUG["LESS_WINDOWS"]:
            window_data = window_data[:3]
        elif self.DEBUG["ONE_WINDOW"]:
            window_data = window_data[10:11]
        return window_data
    
    
    def _update_past_kernels(self, fit_result, first_window, w, kernel):
        if first_window:
            fit_result.all_kernels[:] = kernel
        else:
            fit_result.all_kernels[w.train_start :] = kernel


    def _find_past_kernels(self, fit_result, first_window, w):
        past_kernels = None
        if not first_window and self.model_config.variable_kernels:
            past_kernels = fit_result.all_kernels[w.train_start : w.train_start + self.model_config.los_cutoff]
        return past_kernels

    def fit(self):
        all_fit_results = self.all_fit_results
        
        # --- Main loop ---
        for distro in self.distributions:
            print(f"Distro: {distro}")
            all_fit_results[distro] = self.fit_distro(distro)

        all_fit_results.bake()

        for distro, fit_result in all_fit_results.items():
            train_mean = fit_result.train_relative_errors.mean()
            test_mean = fit_result.test_relative_errors.mean()
            print(f"{distro[:7]}\t Mean Train Error: {float(train_mean):.2f}, Mean Test Error: {float(test_mean):.2f}")
        return self.window_data, all_fit_results

    def fit_distro(self,distro):
        model_config = self.model_config
        series_data = self.series_data

        fit_result = SeriesFitResult(distro)
        fit_result.all_kernels = np.zeros((self.series_data.n_days, self.model_config.kernel_width))

        failed_windows = []
        is_first_window = True
        
        # compartmental models always uses its own fitter
        for window_id, window_info, train_data, test_data in tqdm(self.window_data):
            w = window_info
                
            try:
                if distro == "compartmental":
                    result_obj = fit_compartmental(
                        train_data,
                        test_data,
                        initial_guess_comp=[1/7, 1, 0],
                        los_cutoff=model_config.los_cutoff,
                    )
                    y_pred = calc_its_comp(series_data.x_full, *result_obj.model_config, series_data.y_full[0])
                else:
                    init_vals = self.init_parameters.get(distro)
                    if self.model_config.reuse_last_parametrization:
                        init_vals = self._find_last_valid_parametrization(fit_result, window_id,init_vals)
                    past_kernels = self._find_past_kernels(fit_result, is_first_window, w)

                    result_obj = fit_convolution(
                        distro, train_data, test_data,
                        self.model_config.kernel_width, self.model_config.los_cutoff,
                        distro_init_params=init_vals,
                        past_kernels=past_kernels,
                        error_fun=model_config.error_fun,
                    )

                    self._update_past_kernels(fit_result, is_first_window, w, result_obj.kernel)
                    y_pred = calc_its_convolution(
                        series_data.x_full, fit_result.all_kernels, self.model_config.los_cutoff
                    )
                    
                rel_err = np.abs(y_pred - series_data.y_full) / (series_data.y_full + 1)
                result_obj.rel_train_error = np.mean(rel_err[w.train_window])
                result_obj.rel_test_error = np.mean(rel_err[w.test_window])

            except Exception as e:
                print(f"\tError fitting {distro}: {e}")
                result_obj = SingleFitResult()
                raise e
                
            if not result_obj.success:
                failed_windows.append(window_id)
            fit_result.append(window_info,result_obj)

            is_first_window = False
        if failed_windows:
            print(f"Failed windows for {distro}: {failed_windows}")
        return fit_result

    def _find_last_valid_parametrization(self, fit_result, window_id, init_vals):
        for prev in reversed(fit_result[:window_id]):
            if not prev:
                continue
            return prev.model_config
        return init_vals

    