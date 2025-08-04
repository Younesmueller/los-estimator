import numpy as np
from numba import njit

class ErrorType:
    """Enum for available error function types."""
    MSE = "mse"
    WEIGHTED_MSE = "weighted_mse"



class _ErrorFunctions:
    """Collection of error functions for model fitting."""
    
    @njit
    def weighted_mse(x,y):
        le = len(x)
        weights = np.exp(np.linspace(0,2,le))
        weights /= weights.sum()
        return np.sum(((x - y) ** 2)*weights)

    @njit
    def mse(x, y):
        return np.mean((x - y) ** 2)
    
    errors = {
        ErrorType.MSE: mse,
        ErrorType.WEIGHTED_MSE: weighted_mse
    }
    def __getitem__(self,error_fun):
        """Returns the appropriate error function based on the input string."""
        if error_fun in self.errors:
            return self.errors[error_fun]
        else:
            raise ValueError(f"Unknown Error Function: {error_fun}")
        
ErrorFunctions = _ErrorFunctions()