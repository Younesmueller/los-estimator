"""Core data classes and structures for LOS estimation."""
import numpy as np

import functools
import matplotlib.pyplot as plt



__all__ = [
    "WindowInfo",
    "SeriesData",    
]


class WindowInfo:
    def __init__(self,window,model_config):
        self.window = window        
        self.train_end = self.window
        self.train_start = self.window - model_config.train_width
        self.train_los_cutoff = self.train_start + model_config.los_cutoff
        self.test_start = self.train_end
        self.test_end = self.test_start + model_config.test_width
        
        self.train_window = slice(self.train_start,self.train_end)
        self.train_test_window = slice(self.train_start,self.test_end)
        self.test_window = slice(self.test_start,self.test_end)

        self.model_config = model_config
        
    def __repr__(self):
        return f"WindowInfo(window={self.window}, train_start={self.train_start}, train_end={self.train_end}, test_start={self.test_start}, test_end={self.test_end})"


class SeriesData:
    def __init__(self,x_full,y_full,model_config):
        self.model_config = model_config
        self.x_full = x_full
        self.y_full = y_full
        self._calc_windows(model_config)
        self.n_days = len(self.x_full)

    def _calc_windows(self,model_config):
        start =  model_config.train_width
        self.windows = np.arange(start,len(self.x_full)-model_config.kernel_width, model_config.step)
        self.window_infos = [WindowInfo(window,model_config) for window in self.windows]
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
        return f"SeriesData(n_windows={self.n_windows}, kernel_width={self.model_config.kernel_width}, los_cutoff={self.model_config.los_cutoff})"
        


