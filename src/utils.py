"""
utils.py
--------
Shared utilities: logging setup, directory creation, timing.
"""

import os
import logging
import time
from functools import wraps


def setup_logger(name: str = "framework", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.
    Uses StreamHandler so output appears directly in the terminal.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s  [%(levelname)s]  %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dirs(*paths: str) -> None:
    """Create directories (including nested) if they do not already exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def timer(logger: logging.Logger = None):
    """
    Decorator that logs the wall-clock time taken by a function.
    Usage:
        @timer(logger)
        def expensive_function(): ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            msg = f"{func.__name__} completed in {elapsed:.2f}s"
            if logger:
                logger.info(msg)
            else:
                print(f"[TIMER] {msg}")
            return result
        return wrapper
    return decorator


def safe_divide(numerator, denominator, fill_value=0.0):
    """
    Element-wise division that replaces division-by-zero with fill_value.
    Works on scalars and numpy arrays.
    """
    import numpy as np
    denom = np.where(denominator == 0, np.nan, denominator)
    result = numerator / denom
    return np.where(np.isnan(result), fill_value, result)
