"""Utilities for preparing time-series windows."""
import numpy as np

from ._validation import positive_integer


def segment_data(X: np.ndarray, window_size: int, step_size: int = None) -> np.ndarray:
    """Return full windows as (n_segments, n_channels, window_size).

    step_size defaults to window_size. Partial trailing windows are discarded.
    Raise ValueError when no complete window fits or sizes are not positive integers.
    """
    X = np.asarray(X)
    if X.ndim != 2 or any(size == 0 for size in X.shape):
        raise ValueError('Input must be nonempty 2D (n_channels, n_time)')
    window_size = positive_integer(window_size, 'window_size')
    step_size = window_size if step_size is None else positive_integer(step_size, 'step_size')
    if window_size > X.shape[1]:
        raise ValueError('window_size must not exceed the time length')
    return np.stack([X[:, start:start+window_size]
                     for start in range(0, X.shape[1]-window_size+1, step_size)])
