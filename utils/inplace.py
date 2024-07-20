from collections.abc import Callable
from typing import Any
import warnings


def inplace(f: Callable) -> Callable:
    """
    This function returns new version of data that passed in this function.
    """
    def _f(b: Any) -> Any:
        f(b)
        return b
    return _f

def inplace_windows(f: Callable) -> Callable:
    """
    it is not supported on Windows. 
    The reason is that multiprocessing lib doesn’t have it implemented on Windows. 
    There are some alternatives like dill that can pickle more objects
    """
    warnings.warn("Inplace doesn't work on windows")
    return f