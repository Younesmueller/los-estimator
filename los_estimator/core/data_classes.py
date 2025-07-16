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
        


class Utils:
    def select_series(df, params):
        if params.fit_admissions:
            col = "new_icu_smooth" if params.smooth_data else "new_icu"
        else:
            col = "AnzahlFall" if params.smooth_data else "daily"
        return df[col].values, df["icu"].values

