import numpy as np
import torch
import warnings

def check_backend(device: str = 'cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU.")
        return torch.device('cpu')
    return torch.device(device)

def compute_mvgc(X: np.ndarray, p: int, alpha: float = 0.01, device: str = 'cuda') -> np.ndarray:
    """
    计算 MVGC 矩阵（用户主入口）。
    
    Args:
        X: (n_channels, n_time) 时间序列数据
        p: VAR 滞后阶数 (Lag order)
        alpha: 岭回归正则化系数 (Ridge regularization)，解决 NaN 的关键！
               建议值 0.01 ~ 0.1。如果数据很干净，可以用 1e-4。
        device: 'cuda' 或 'cpu'
        
    Returns:
        F: (n_channels, n_channels) 格兰杰因果矩阵。F[i, j] 代表 j -> i 的因果强度。
    """
    X = np.asarray(X)
    
    # 维度检查与修正
    if X.ndim != 2:
        raise ValueError(f"Input must be 2D (n_channels, n_time), got {X.shape}")
    
    n_channels, n_time = X.shape
    if n_channels > n_time:
        warnings.warn(f"Warning: Channels ({n_channels}) > Time points ({n_time}). Did you transpose the input?")

    # 视作 Batch = 1 进行处理
    X_batch = X[np.newaxis, :, :] # (1, n, T)
    
    F_batch = compute_mvgc_batch(X_batch, p, alpha, device)
    
    return F_batch[0] # 返回 (n, n)

def compute_mvgc_batch(X_batch: np.ndarray, p: int, alpha: float = 0.01, device: str = 'cuda') -> np.ndarray:
    """
    批量 MVGC 计算函数。
    """
    dev = check_backend(device)
    X_t = torch.from_numpy(X_batch).to(dev, dtype=torch.float64) # 使用 float64 提高精度
    
    # 调用内部 Torch 内核
    with torch.no_grad():
        F_res = _mvgc_batch_kernel(X_t, p, alpha, dev)
        
    return F_res.cpu().numpy()

def _mvgc_batch_kernel(segments: torch.Tensor, p: int, alpha: float, dev: torch.device) -> torch.Tensor:
    """
    PyTorch 核心计算逻辑：向量化 OLS 回归 + 岭回归正则化
    segments: (B, n, T)
    return: (B, n, n)
    """
    B, n, T = segments.shape
    if T <= p:
        raise ValueError(f'Time length {T} must be > order {p}')
        
    # 1. 去均值 (Centering)
    seg_t = segments - segments.mean(dim=2, keepdim=True)
    
    # 2. 构建 Y (响应变量) 和 Z (设计矩阵)
    # Y: (B, N, n), N = T-p
    # Z: (B, N, n*p)
    N = T - p
    Y = seg_t[:, :, p:].permute(0, 2, 1) 
    
    Z_list = []
    for k in range(1, p + 1):
        # 滞后 k 个时间点
        lag = seg_t[:, :, p - k : T - k].permute(0, 2, 1)
        Z_list.append(lag)
    Zs = torch.cat(Z_list, dim=2) # (B, N, n*p)
    
    D = n * p # 特征维度
    
    # 3. 计算 Full Model (全模型) 的残差方差
    # 求解 Y = Z * Beta + E
    # Beta = (Z^T Z + alpha*I)^-1 Z^T Y
    
    ZsT = Zs.permute(0, 2, 1) # (B, D, N)
    ZTZ = torch.bmm(ZsT, Zs)  # (B, D, D)
    ZTY = torch.bmm(ZsT, Y)   # (B, D, n)
    
    # === 关键：正则化 (Ridge) 防止奇异矩阵导致的 NaN ===
    # 添加 alpha 到对角线
    eye = torch.eye(D, device=dev, dtype=ZTZ.dtype).unsqueeze(0).expand(B, D, D)
    ZTZ_reg = ZTZ + alpha * eye
    
    # 使用 cholesky 或 solve 求解比求逆更稳定
    # beta_full: (B, D, n)
    try:
        # 尝试更快的 Cholesky
        L = torch.linalg.cholesky(ZTZ_reg)
        beta_full = torch.cholesky_solve(ZTY, L)
    except:
        # 回退到标准求逆 (伪逆)
        beta_full = torch.linalg.solve(ZTZ_reg, ZTY)

    # 计算 Full Model 残差方差 sigma_full
    resid_full = Y - torch.bmm(Zs, beta_full)
    # 只需要对角线部分的方差 (Generalized Variance 近似为变量方差的乘积，在MVGC中简化为逐变量比较)
    sigma_full = resid_full.pow(2).sum(dim=1) # (B, n)
    
    # 4. 计算 Reduced Models (逐个剔除源变量 j)
    # 结果矩阵 F (B, target, source)
    F_batch = torch.zeros((B, n, n), dtype=torch.float64, device=dev)
    
    # 预先生成所有列的索引
    all_cols = torch.arange(D, device=dev)
    
    for j in range(n): # j 是 Source (因)
        # 剔除 j 的所有滞后项 (columns: j*p, j*p+1, ..., (j+1)*p-1)
        # 保留 Target i 和其他变量 Z
        
        # 确定要保留的列索引
        # 这是一个简单的掩码操作
        mask = torch.ones(D, dtype=torch.bool, device=dev)
        mask[j*p : (j+1)*p] = False
        keep_idx = all_cols[mask]
        
        # 提取 Reduced 设计矩阵 Z_red
        Zs_red = Zs.index_select(2, keep_idx) # (B, N, D_reduced)
        D_red = Zs_red.shape[2]
        
        # OLS for Reduced Model
        Zs_red_T = Zs_red.permute(0, 2, 1)
        ZTZ_r = torch.bmm(Zs_red_T, Zs_red)
        ZTY_r = torch.bmm(Zs_red_T, Y)
        
        # Regularization again
        eye_r = torch.eye(D_red, device=dev, dtype=ZTZ_r.dtype).unsqueeze(0).expand(B, D_red, D_red)
        ZTZ_r_reg = ZTZ_r + alpha * eye_r
        
        try:
            L_r = torch.linalg.cholesky(ZTZ_r_reg)
            beta_r = torch.cholesky_solve(ZTY_r, L_r)
        except:
            beta_r = torch.linalg.solve(ZTZ_r_reg, ZTY_r)
            
        resid_r = Y - torch.bmm(Zs_red, beta_r)
        sigma_red = resid_r.pow(2).sum(dim=1) # (B, n) - 每个变量作为 Target 时的 Reduced 方差
        
        # 5. 计算 Granger Causality
        # F_{j->i} = ln(var_reduced_i / var_full_i)
        # 注意：这里 sigma_red 和 sigma_full 都是 (B, n)，表示 n 个变量作为 Target 时的方差
        # 当前循环 j 是 Source。
        # 所以 ratio[b, i] 代表：Source j 被剔除后，Target i 的方差变化
        
        # 加上极小值防止除以0
        ratio = (sigma_red + 1e-12) / (sigma_full + 1e-12)
        
        # 结果填入 F 的第 j 列 (j -> all targets)
        F_batch[:, :, j] = torch.log(ratio)
        
    # 6. 后处理：对角线置 0 (无自因果)
    idx = torch.arange(n, device=dev)
    F_batch[:, idx, idx] = 0.0
    
    # 修正可能的负值 (由于数值误差，完全无关时 log 可能出现微小负值)
    F_batch = torch.relu(F_batch)
    
    return F_batch