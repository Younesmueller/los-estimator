from typing import Callable, Dict, List, Iterable, Optional, Tuple
from los_estimator.fitting.errors import ErrorFunctions
import pandas as pd


class EvaluationResult:
    def __init__(
        self,
        distros: List[str],
        metric_names: List[str],
        train: Optional[np.ndarray] = None,
        test: Optional[np.ndarray] = None,
        window_data_package: Optional[List] = None,
    ):
        self.train = train
        self.test = test
        self.distros = distros
        self.metric_names = metric_names
        self.window_data_package = window_data_package

    def iter_distros(self, ret_arr=True) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
        for i_distro, distro in enumerate(self.distros):
            if ret_arr:
                yield i_distro, distro, self.train[i_distro, :, :], self.test[i_distro, :, :]
            else:
                yield i_distro, distro

    def iter_metrics(self, ret_arr=True) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
        for i_metric, metric_name in enumerate(self.metric_names):
            if ret_arr:
                yield metric_name, self.train[:, :, i_metric], self.test[:, :, i_metric]
            else:
                yield i_metric, metric_name

    def by_metric(
        self, metric_name: Optional[str], metric_index: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if metric_index is None:
            metric_index = self.metric_names.index(metric_name)
        return self.train[:, :, metric_index], self.test[:, :, metric_index]

    def by_distro(
        self, distro_name: Optional[str], distro_index: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if distro_index is None:
            distro_index = self.distros.index(distro_name)
        return self.train[distro_index, :, :], self.test[distro_index, :, :]

    def by_distro_and_metric(
        self,
        distro_name: Optional[str],
        metric_name: Optional[str],
        distro_index: Optional[int] = None,
        metric_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if distro_index is None:
            distro_index = self.distros.index(distro_name)
        if metric_index is None:
            metric_index = self.metric_names.index(metric_name)
        return self.train[distro_index, :, metric_index], self.test[distro_index, :, metric_index]


class WindowDataPackage:
    def __init__(self, all_fit_results: Dict[str, any], series_data: SeriesData):
        self.all_fit_results = all_fit_results
        self.series_data = series_data
        self.data = None
        self.build_package()

    def iterate_index(self):
        for i_distro in range(self.n_distros):
            for i_window in range(self.n_windows):
                yield i_distro, i_window

    def build_package(self):
        self.n_distros = len(self.all_fit_results)
        self.n_windows = self.all_fit_results.n_windows
        data = np.empty((self.n_distros, self.n_windows), dtype=object)

        for i_distro, fit_result in enumerate(self.all_fit_results.values()):
            for i_window, (single_fit_result, w) in enumerate(zip(fit_result.fit_results, fit_result.window_infos)):
                y_pred_train = single_fit_result.train_prediction[w.kernel_width :]
                y_pred_test = single_fit_result.test_prediction[w.kernel_width :]

                y_true_train = self.series_data.y_full[w.training_prediction_start : w.train_end]
                y_true_test = self.series_data.y_full[w.test_start : w.test_end]
                x_train = np.arange(w.training_prediction_start, w.train_end)
                x_test = np.arange(w.test_start, w.test_end)

                data[i_distro][i_window] = (y_true_train, y_pred_train, y_true_test, y_pred_test, x_train, x_test, w)

        self.data = data


class Evaluator:
    def __init__(
        self,
        all_fit_results: Dict[str, any],
        series_data: SeriesData,
        metrics: Iterable[Tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]],
    ):
        self.all_fit_results = all_fit_results
        self.series_data = series_data
        self.metric_names = [m[0] for m in metrics]
        self.metric_functions = [m[1] for m in metrics]
        self.result: Optional[EvaluationResult] = EvaluationResult(
            distros=list(all_fit_results.keys()), metric_names=self.metric_names
        )
        self.window_data_package = WindowDataPackage(self.all_fit_results, self.series_data)

    def calculate_metrics(
        self,
    ):
        n_distros = len(self.all_fit_results.distros)
        n_windows = self.window_data_package.n_windows
        res_train = np.zeros((n_distros, n_windows, len(self.metric_names)))
        res_test = np.zeros((n_distros, n_windows, len(self.metric_names)))
        for i_distro, i_window in self.window_data_package.iterate_index():
            (y_true_train, y_pred_train, y_true_test, y_pred_test, *_) = self.window_data_package.data[i_distro][
                i_window
            ]

            for i_metric, metric_func in enumerate(self.metric_functions):
                res_train[i_distro, i_window, i_metric] = metric_func(y_true_train, y_pred_train)
                res_test[i_distro, i_window, i_metric] = metric_func(y_true_test, y_pred_test)
        self.result.train = res_train
        self.result.test = res_test
        return self.result

    def save_result(self, path: str):

        distros = self.all_fit_results.distros
        metrics = self.metric_names
        windows = self.series_data.windows

        arr_train = self.result.train
        arr_test = self.result.test
        n_dists, n_days, n_metrics = arr_train.shape

        df_train = (
            pd.DataFrame(arr_train.reshape(n_dists * n_days, n_metrics), columns=metrics)
            .assign(distribution=np.repeat(distros, n_days), day=np.tile(windows, n_dists))
            .melt(id_vars=["distribution", "day"], var_name="metric", value_name="value")
        )
        df_test = (
            pd.DataFrame(arr_test.reshape(n_dists * n_days, n_metrics), columns=metrics)
            .assign(distribution=np.repeat(distros, n_days), day=np.tile(windows, n_dists))
            .melt(id_vars=["distribution", "day"], var_name="metric", value_name="value")
        )

        df_train = df_train[["metric", "day", "distribution", "value"]]
        df_test = df_test[["metric", "day", "distribution", "value"]]

        df_train.to_csv(path + "/metrics_train.csv", index=False)
        df_test.to_csv(path + "/metrics_test.csv", index=False)
        return df_train, df_test


metrics_over_time: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "absolute_error": lambda t, p: np.abs(t - p),
    "squared_error": lambda t, p: np.square(t - p),
    "relative_error": lambda t, p: np.abs(t - p) / (t + 1e-8),
    "inc_error": lambda t, p: np.abs(t - p) * np.abs(t),
}
