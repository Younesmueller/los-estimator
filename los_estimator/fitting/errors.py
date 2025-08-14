import numpy as np
from numba import njit

__all__ = [
    "ErrorType",
    "ErrorFunctions",
]


class ErrorType:
    """Enum for available error function types."""

    MSE = "mse"
    WEIGHTED_MSE = "weighted_mse"
    MAE = "mae"
    RMSE = "rmse"
    MAPE = "mape"
    SMAPE = "smape"
    R2 = "r2"
    INC_ERROR = "inc_error"


class _ErrorFunctions:
    """Collection of error functions for model fitting."""

    def cap_err(y_true, y_pred, cap, a=0.02):
        weights = np.exp(((y_true - cap) / cap) * a)
        weights = np.abs((y_true - cap) / cap)
        weights = y_true.copy()
        weights /= weights.sum()
        return weights * np.abs(y_true - y_pred)

    @njit
    def inc_error(y_true, y_pred):
        weights = y_true / y_true.sum()
        return weights * np.abs(y_true - y_pred)

    @njit
    def weighted_mse(x, y):
        le = len(x)
        weights = np.exp(np.linspace(0, 2, le))
        weights /= weights.sum()
        return np.sum(((x - y) ** 2) * weights)

    @njit
    def mse(x, y):
        return np.mean((x - y) ** 2)

    @njit
    def mae(x, y):
        return np.mean(np.abs(x - y))

    @njit
    def rmse(x, y):
        return np.sqrt(np.mean((x - y) ** 2))

    @njit
    def mape(x, y):
        return np.mean(np.abs((x - y) / x)) * 100

    @njit
    def smape(x, y):
        return np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y)) * 2) * 100

    @njit
    def r2(x, y):
        ss_res = np.sum((x - y) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2)
        return 1 - (ss_res / ss_tot)

    errors = {
        ErrorType.MSE: mse,
        ErrorType.WEIGHTED_MSE: weighted_mse,
        ErrorType.MAE: mae,
        ErrorType.RMSE: rmse,
        ErrorType.MAPE: mape,
        ErrorType.SMAPE: smape,
        ErrorType.R2: r2,
    }

    def __getitem__(self, error_fun):
        """Returns the appropriate error function based on the input string."""
        if error_fun in self.errors:
            return self.errors[error_fun]
        else:
            raise ValueError(f"Unknown Error Function: {error_fun}")


ErrorFunctions = _ErrorFunctions()
