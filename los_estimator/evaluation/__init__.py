"""Evaluation metrics and tools for model assessment.

This module provides comprehensive evaluation capabilities for length of stay
models, including various error metrics and statistical measures.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..fitting.errors import ErrorFunctions

__all__ = [
    "FitResultEvaluator",
]


metrics_over_time: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "absolute_error": lambda t, p: np.abs(t - p),
    "squared_error": lambda t, p: np.square(t - p),
    "relative_error": lambda t, p: np.abs(t - p) / (t + 1e-8),
    "inc_error": lambda t, p: np.abs(t - p) * np.abs(t),
}


class FitResultEvaluator:
    """Evaluator for model fit results and predictions.

    Provides comprehensive evaluation of model performance using various
    error metrics and statistical measures, both as aggregated values
    and time series.

    Attributes:
        distro (str): Distribution type being evaluated.
        y_true (np.ndarray): True/observed values.
        y_pred (np.ndarray): Predicted values.
        metrics (np.ndarray): Computed aggregated metrics.
        metrics_over_time (np.ndarray): Time-series metrics.
    """

    def __init__(
        self,
        distro: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Initialize the fit result evaluator.

        Args:
            distro (str): Name of the distribution/model being evaluated.
            y_true (np.ndarray): Array of true/observed values.
            y_pred (np.ndarray): Array of predicted values.
        """
        self.distro: str = distro
        self.y_true: np.ndarray = y_true
        self.y_pred: np.ndarray = y_pred
        self._metrics_over_time: List[str] = [
            "absolute_error",
            "squared_error",
            "relative_error",
            "inc_error",
        ]
        self._metrics: List[str] = list(ErrorFunctions.errors.keys())
        self.metrics: np.ndarray
        self.metrics_over_time: np.ndarray

    def _get_data(self, y_true: Optional[np.ndarray], y_pred: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Get data arrays for evaluation.

        Uses provided arrays or falls back to instance attributes.

        Args:
            y_true (Optional[np.ndarray]): True values array.
            y_pred (Optional[np.ndarray]): Predicted values array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (y_true, y_pred) arrays to use.
        """
        y_true_out = y_true if y_true is not None else self.y_true
        y_pred_out = y_pred if y_pred is not None else self.y_pred

        # y_true_out: np.ndarray = self.y_true
        # y_pred_out: np.ndarray = self.y_pred

        # if y_true is not None:
        #     y_true_out = y_true
        # if y_pred is not None:
        #     y_pred_out = y_pred
        return y_true_out, y_pred_out

    def evaluate(
        self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate model performance using various metrics.

        Computes both aggregated metrics and time-series metrics for
        comprehensive model assessment.

        Args:
            y_true (Optional[np.ndarray], optional): True values. Uses instance
                attribute if None.
            y_pred (Optional[np.ndarray], optional): Predicted values. Uses
                instance attribute if None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (metrics, metrics_over_time) arrays.
        """
        y_true, y_pred = self._get_data(y_true, y_pred)

        self.metrics = np.empty((len(self._metrics),), dtype=float)
        self.metrics_over_time = np.empty((len(self._metrics_over_time), len(y_true)), dtype=float)

        for i, name in enumerate(self._metrics):
            func = ErrorFunctions[name]
            self.metrics[i] = func(y_true, y_pred)

        for i, name in enumerate(self._metrics_over_time):
            func = metrics_over_time[name]
            self.metrics_over_time[i] = func(y_true, y_pred)
        return self.metrics, self.metrics_over_time

    def save(self, path: str):
        df = pd.DataFrame(self.metrics[np.newaxis], columns=self._metrics, index=[self.distro])
        df.to_csv(path + f"/{self.distro}_metrics.csv")
        df = pd.DataFrame(self.metrics_over_time.T, columns=self._metrics_over_time)
        df.to_csv(path + f"/{self.distro}_metrics_over_time.csv")

    def __repr__(self):
        return f"FitResultEvaluator(distro={self.distro}, errors={list(zip(self._metrics,self.metrics))})"
