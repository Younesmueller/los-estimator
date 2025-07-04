from typing import OrderedDict
import numpy as np
import pandas as pd
import types

import functools
import matplotlib.pyplot as plt



class Params (types.SimpleNamespace):
    pass

class VisualizationContext(types.SimpleNamespace):
    pass

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


class SeriesData:
    def __init__(self,df_occupancy,params,new_icu_day):
        self.params = params
        self.new_icu_day = new_icu_day
        self.x_full,self.y_full = Utils.select_series(df_occupancy, params)
        self._calc_windows(params)
        self.n_days = len(self.x_full)

    def _calc_windows(self,params):
        start = 0
        if params.fit_admissions:
            start = self.new_icu_day + params.train_width
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
        return f"SeriesData(n_windows={self.n_windows}, kernel_width={self.params.kernel_width}, los_cutoff={self.params.los_cutoff})"
        

class SeriesFitResult:
    def __init__(self, distro):
        self.distro = distro
        self.window_infos = []
        self.fit_results = []
        self.train_relative_errors = None
        self.test_relative_errors = None        
        self.successes = []
        self.n_success = np.nan
        self.all_kernels: np.ndarray = None

    def append(self, window_info, fit_result):
        # if not isinstance(window_info, WindowInfo):
        #     raise TypeError("window_info must be an instance of WindowInfo")
        # if not isinstance(fit_result, SingleFitResult):
        #     raise TypeError("fit_result must be an instance of SingleFitResult")
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
    def __init__(self, distros=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if distros is not None:
            for distro in distros:
                self[distro] = SeriesFitResult(distro)
            self.distros = list(self.keys())
            self.results = list(self.values())

    def bake(self):
        self.distros = list(self.keys())
        self.results = list(self.values())

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
        df_train = pd.DataFrame(self.train_errors_by_distro, columns=self.distros)
        df_test = pd.DataFrame(self.test_errors_by_distro, columns=self.distros)

        # Compute mean finite loss and failure rate for each model
        summary = pd.DataFrame(index=self.distros)
        summary["Failure Rate"] = self.failures_by_distro.mean(axis=0)

        summary["Mean Loss Train"] = df_train.replace(np.inf, np.nan).mean()
        summary["Median Loss Train"] = df_train.replace(np.inf, np.nan).median()
        summary["Upper Quartile Train"] = df_train.quantile(0.75)
        summary["Lower Quartile Train"] = df_train.quantile(0.25)

        summary["Mean Loss Test"] = df_test.replace(np.inf, np.nan).mean()
        summary["Median Loss Test"] = df_test.replace(np.inf, np.nan).median()

        def remove_outliers(df, col):
            summary[col] = np.nan
            for distro in self.distros:
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




class Utils:
    def select_series(df, params):
        if params.fit_admissions:
            col = "new_icu_smooth" if params.smooth_data else "new_icu"
        else:
            col = "AnzahlFall" if params.smooth_data else "daily"
        return df[col].values, df["icu"].values


class SingleFitResult:
    def __init__(self, 
        distro=None,
        train_data=None,
        test_data=None,
        success=None,
        minimization_result=None,
        train_error=None,
        test_error=None,
        rel_train_error=None,
        rel_test_error=None,
        kernel=None,
        curve=None,
        params=None
    ):        
        self.distro = distro
        self.train_data = train_data
        self.test_data = test_data
        self.success = success or False
        self.minimization_result = minimization_result
        self.train_error = train_error
        self.test_error = test_error
        self.rel_train_error = rel_train_error
        self.rel_test_error = rel_test_error
        self.kernel = kernel
        self.curve = curve
        self.params = params #TODO: Split in Curve params and distro params
    def __repr__(self):
        # return a string with all variables
        if self is None:
            return None
        return (f"SingleFitResult(distro={self.distro}, "
                f"success={self.success}, "
                f"train_error={self.train_error}, "
                f"test_error={self.test_error}, "
                f"rel_train_error={self.rel_train_error}, "
                f"rel_test_error={self.rel_test_error}, "
                f"kernel={self.kernel.shape}, "
                f"curve={self.curve.shape}, "
                f"params={self.params})")