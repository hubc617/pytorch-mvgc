import numpy as np
import warnings

def compute_mvgc_cpu(X: np.ndarray, p: int, alpha: float = 0.01) -> np.ndarray:
    """
    CPU implementation of MVGC using NumPy (OLS with Ridge Regression).
    Acts as a fallback or reference implementation.
    
    Args:
        X: (n_channels, n_time)
        p: Lag order
        alpha: Ridge regularization parameter
    """
    X = np.asarray(X)
    n, T = X.shape
    if T <= p:
        raise ValueError(f'Time length {T} must be > order {p}')
        
    # 1. Prepare Data
    # Center the data
    X_mean = X - X.mean(axis=1, keepdims=True)
    
    # Build Y (Target) and Z (Design Matrix)
    # Y: (n, N) where N = T - p
    N = T - p
    Y = X_mean[:, p:]
    
    Z_list = []
    for k in range(1, p + 1):
        lag = X_mean[:, p-k : T-k]
        Z_list.append(lag)
    # Stack lags: Z shape (n*p, N)
    Z = np.vstack(Z_list)
    
    # 2. Full Model Regression
    # Beta = (Z Z.T + alpha I)^-1 Z Y.T
    # We solve (Z Z.T + alpha I) Beta = Z Y.T
    
    ZZT = Z @ Z.T
    ZYT = Z @ Y.T
    
    # Ridge Regularization
    D = n * p
    ZZT_reg = ZZT + alpha * np.eye(D)
    
    # Solve linear system
    try:
        # utilize symmetry for speed (equivalent to cholesky solve)
        beta_full = np.linalg.solve(ZZT_reg, ZYT)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if somehow still singular
        beta_full = np.linalg.pinv(ZZT_reg) @ ZYT
        
    # Residuals
    # Note: Beta shape is (n*p, n), so Z.T @ Beta gives (N, n), transpose to (n, N)
    preds_full = beta_full.T @ Z 
    res_full = Y - preds_full
    # Variance of residuals (diagonal of covariance matrix)
    sig_full = np.var(res_full, axis=1, ddof=0)
    
    # 3. Reduced Models (Loop over source j)
    F = np.zeros((n, n))
    
    # Cache ZZT parts to avoid recomputing? 
    # For CPU simple implementation, we just slice Z.
    
    for j in range(n):
        # We want to calculate causality j -> i (for all i)
        # So we remove source j from the predictors Z
        
        # Identify rows in Z to keep (remove lag rows corresponding to channel j)
        # Z rows: 0..n-1 (lag 1), n..2n-1 (lag 2)...
        keep_indices = []
        for k in range(p):
            # The row index for channel j at lag k+1 is: j + k*n
            # Wait, construction of Z above:
            # Z_list[0] is shape (n, N). It contains ch0_lag1, ch1_lag1, ...
            # So Z structure is: [ch0_l1, ch1_l1, ..., ch0_l2, ...].T
            # Correct index for channel c at lag k (1-based) is: c + (k-1)*n
            
            # Let's verify standard flattening. 
            # Z = vstack(Z_list). Z_list[0] has n rows.
            # Row x corresponds to channel (x % n)
            pass

        # Easier way: mask indices where (idx % n) != j
        all_indices = np.arange(D)
        mask = (all_indices % n) != j
        idx_reduced = all_indices[mask]
        
        Z_red = Z[idx_reduced, :]
        D_red = len(idx_reduced)
        
        # OLS Reduced
        ZZT_r = Z_red @ Z_red.T
        ZYT_r = Z_red @ Y.T
        
        ZZT_r_reg = ZZT_r + alpha * np.eye(D_red)
        
        try:
            beta_r = np.linalg.solve(ZZT_r_reg, ZYT_r)
        except np.linalg.LinAlgError:
            beta_r = np.linalg.pinv(ZZT_r_reg) @ ZYT_r
            
        preds_r = beta_r.T @ Z_red
        res_r = Y - preds_r
        sig_r = np.var(res_r, axis=1, ddof=0)
        
        # 4. Granger Causality
        # F_{j->i} = ln(sig_r_i / sig_full_i)
        ratio = (sig_r + 1e-12) / (sig_full + 1e-12)
        F[:, j] = np.log(ratio)
        
    # Zero diagonal
    np.fill_diagonal(F, 0.0)
    
    # Clean up small negative values
    F[F < 0] = 0.0
    
    return F