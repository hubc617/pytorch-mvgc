"""Shared input checks for the NumPy and PyTorch engines."""
from numbers import Integral, Real

import numpy as np


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_data(data, ndim):
    array = np.asarray(data)
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        shape = "(n_channels, n_time)" if ndim == 2 else "(batch, n_channels, n_time)"
        raise ValueError(f"Input must be nonempty {ndim}D {shape}, got {array.shape}")
    if array.dtype.kind not in "iuf":
        raise ValueError("Input must contain real numeric values")
    # A native contiguous copy also supports negative strides and integer input.
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Input must contain only finite values")
    return array


def validate_parameters(p, alpha, n_time):
    p = positive_integer(p, "p (lag order)")
    if p >= n_time:
        raise ValueError(f"Time length {n_time} must be > order p={p}")
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("alpha must be a finite nonnegative real number")
    return p, float(alpha)
