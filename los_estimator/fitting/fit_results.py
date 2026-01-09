from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from los_estimator.config import ModelConfig
from los_estimator.core import WindowInfo


@dataclass
class SingleFitResult:
    distro: str
    train_data: object
    test_data: object
    success: bool
    minimization_result: dict
    train_error: np.ndarray
    test_error: np.ndarray
    kernel: np.ndarray
    model_config: ModelConfig
    train_prediction: Optional[np.ndarray] = None
    test_prediction: Optional[np.ndarray] = None
    rel_train_error: Optional[np.ndarray] = None
    rel_test_error: Optional[np.ndarray] = None
    curve: Optional[np.ndarray] = None

    def __repr__(self):
        # return a string with all variables
        if self is None:
            return None
        return (
            f"SingleFitResult(distro={self.distro}, "
            f"success={self.success}, "
            f"train_error={self.train_error}, "
            f"test_error={self.test_error}, "
            f"rel_train_error={self.rel_train_error}, "
            f"rel_test_error={self.rel_test_error}, "
            f"kernel={self.kernel.shape}, "
            f"model_config={self.model_config})"
        )


class SeriesFitResult:
    distro: str
    window_infos: list[WindowInfo]
    fit_results: list[SingleFitResult]
    train_errors: np.ndarray
    test_errors: np.ndarray
    successes: list[bool]
    n_success: np.ndarray
    all_kernels: np.ndarray
    transition_rates: np.ndarray
    transition_delays: np.ndarray

    def __init__(self, distro):
        self.distro = distro
        self.window_infos = []
        self.fit_results = []
        self.train_errors = None
        self.test_errors = None
        self.successes = []
        self.n_success = None
        self.all_kernels = None

    def append(self, window_info, fit_result):
        self.window_infos.append(window_info)
        self.fit_results.append(fit_result)

    def bake(self):
        self._collect_errors()
        self.successes = [fr.success for fr in self.fit_results]
        self.n_success = sum(self.successes)
        self.transition_rates = np.array(
            [fr.model_config[0] if (fr is not None) else np.nan for fr in self.fit_results]
        )
        self.transition_delays = np.array(
            [fr.model_config[1] if (fr is not None) else np.nan for fr in self.fit_results]
        )

    def _collect_errors(self):
        self.errors_collected = True
        train_err = np.empty(len(self.fit_results))
        test_err = np.empty(len(self.fit_results))
        for i, fr in enumerate(self.fit_results):
            if fr is None:
                train_err[i] = np.inf
                test_err[i] = np.inf
                continue
            train_err[i] = fr.train_error
            test_err[i] = fr.test_error
        self.train_errors = train_err
        self.test_errors = test_err

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
        return f"SeriesFitResult(distro={self.distro}, n_windows={len(self.window_infos)}, train_relative_error: {len(self.train_errors)}, test_relative_error: {len(self.test_errors)})"


class MultiSeriesFitResults(OrderedDict[str, SeriesFitResult]):
    results: list[SeriesFitResult]
    distros: list[str]
    n_windows: int
    train_errors_by_distro: np.ndarray
    test_errors_by_distro: np.ndarray
    successes_by_distro: np.ndarray
    failures_by_distro: np.ndarray
    n_success_by_distro: np.ndarray
    transition_rates_by_distro: np.ndarray
    transition_delays_by_distro: np.ndarray
    summary: pd.DataFrame

    def __init__(self, distros=None, *args, **kwargs):
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
        self.train_errors_by_distro = np.array([fr.train_errors for fr in self.results]).T
        self.test_errors_by_distro = np.array([fr.test_errors for fr in self.results]).T
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
                Q1, Q3 = df[distro].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                # filter out outliers
                mask = (df[distro] < (Q1 - 1.5 * IQR)) | (df[distro] > (Q3 + 1.5 * IQR))
                summary.at[distro, col] = df[distro][~mask].mean()

        remove_outliers(df_test, "Mean Loss Test (no outliers)")
        remove_outliers(df_train, "Mean Loss Train (no outliers)")

        self.summary = summary

    def __repr__(self):
        return f"MultiSeriesFitResults(distros={self.distros}, n_windows={self.n_windows})"
