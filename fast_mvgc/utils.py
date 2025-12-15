# fast_mvgc/utils.py
import numpy as np

def segment_data(X: np.ndarray, window_size: int, step_size: int = None) -> np.ndarray:
    """
    Generic segmentation function.
    Args:
        X: (n_channels, total_time)
        window_size: Length of each segment
        step_size: Stride (default = window_size, i.e., no overlap)
    Returns:
        segments: (n_segments, n_channels, window_size)
    """
    if step_size is None:
        step_size = window_size
        
    n_channels, n_time = X.shape
    starts = range(0, n_time - window_size + 1, step_size)
    segments = [X[:, s:s+window_size] for s in starts]
    return np.stack(segments, axis=0)