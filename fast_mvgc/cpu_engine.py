"""NumPy reference engine for finite-order conditional Granger regression."""
import numpy as np

from ._validation import validate_data, validate_parameters


def compute_mvgc_cpu(X: np.ndarray, p: int, alpha: float = 0.01) -> np.ndarray:
    """Return F[target, source] using NumPy ridge regression.

    X has shape (n_channels, n_time). alpha must be finite and nonnegative;
    alpha=0 uses least squares and supports rank-deficient designs.
    """
    X = validate_data(X, ndim=2)
    n, T = X.shape
    p, alpha = validate_parameters(p, alpha, T)
    if n == 1:
        return np.zeros((1, 1), dtype=np.float64)
    centered = X - X.mean(axis=1, keepdims=True)
    Y = centered[:, p:].T
    Z = np.concatenate([centered[:, p-k:T-k].T for k in range(1, p+1)], axis=1)

    def residual_sse(design):
        if alpha == 0:
            beta = np.linalg.lstsq(design, Y, rcond=None)[0]
        else:
            gram = design.T @ design + alpha * np.eye(design.shape[1])
            rhs = design.T @ Y
            try:
                beta = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(gram) @ rhs
        return ((Y - design @ beta) ** 2).sum(axis=0)

    full = residual_sse(Z)
    F = np.zeros((n, n), dtype=np.float64)
    columns = np.arange(n * p)
    for source in range(n):
        reduced = residual_sse(Z[:, columns % n != source])
        F[:, source] = np.maximum(np.log((reduced + 1e-12) / (full + 1e-12)), 0)
    np.fill_diagonal(F, 0)
    return F
