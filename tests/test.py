import numpy as np
import torch
from fast_mvgc.core import compute_mvgc # 确保引用了上面的新文件

def test_causality_fix():
    print("=== 开始 MVGC 正确性测试 ===")
    
    # 1. 生成标准的 VAR(1) 3通道模型
    # 0 -> 1 (强因果)
    # 2 独立
    np.random.seed(42)
    n_time = 2000
    X = np.zeros((3, n_time))
    noise = np.random.randn(3, n_time) # 标准正态噪声
    
    print("生成数据中: Node 0 -> Node 1 ...")
    for t in range(1, n_time):
        X[0, t] = 0.8 * X[0, t-1] + noise[0, t]
        # X[1] 依赖 X[0] 的过去值
        X[1, t] = 0.5 * X[1, t-1] + 0.5 * X[0, t-1] + noise[1, t] 
        X[2, t] = 0.7 * X[2, t-1] + noise[2, t]

    # 2. 计算 MVGC
    # 注意：alpha (正则化) 对于数值稳定性至关重要
    try:
        F = compute_mvgc(X, p=1, alpha=0.01, device='cuda')
    except Exception as e:
        print(f"GPU计算失败，尝试CPU... 错误: {e}")
        F = compute_mvgc(X, p=1, alpha=0.01, device='cpu')

    print("\n计算结果 F (F[i, j] 表示 j->i):")
    print(np.round(F, 4))
    
    # 3. 验证
    # 我们期望 F[1, 0] (0->1) 是显著的
    val_0_to_1 = F[1, 0]
    val_1_to_0 = F[0, 1]
    
    print(f"\n关键路径检查:")
    print(f"0 -> 1 (预期 > 0.05): {val_0_to_1:.4f}", "✅ PASS" if val_0_to_1 > 0.05 else "❌ FAIL")
    print(f"1 -> 0 (预期 < 0.05): {val_1_to_0:.4f}", "✅ PASS" if val_1_to_0 < 0.05 else "❌ FAIL (False Positive)")
    
    if np.isnan(F).any():
        print("\n❌ 警告: 结果中包含 NaN!")
    else:
        print("\n✅ 结果数值正常 (无 NaN).")

if __name__ == "__main__":
    test_causality_fix()