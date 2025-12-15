import numpy as np
import time
import os
import sys

# 假设用户在项目根目录下运行，添加路径以便导入库
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fast_mvgc import compute_mvgc_batch
from fast_mvgc.utils import segment_data

def demo_high_dim():
    print("=== 128通道数据 MVGC 计算演示 ===")
    
    # 1. 模拟数据生成 (128通道, 2000时间点)
    # 模拟真实场景：数据可能很长，我们需要切片处理
    n_channels = 128
    n_time = 2000
    segment_len = 200 # 你们项目中使用的窗口长度
    
    print(f"1. 生成随机数据 ({n_channels}x{n_time})...")
    X = np.random.randn(n_channels, n_time).astype(np.float32)
    
    # 2. 切片 (Segmentation)
    print(f"2. 切分数据 (窗口长度={segment_len})...")
    # 生成形状 (Batch, n_channels, segment_len)
    segments = segment_data(X, window_size=segment_len, step_size=segment_len)
    print(f"   生成了 {segments.shape[0]} 个片段，形状: {segments.shape}")
    
    # 3. 批量计算
    print("3. 开始 GPU 批量计算...")
    t0 = time.time()
    
    try:
        # p=5, alpha=0.01
        F_batch = compute_mvgc_batch(segments, p=5, alpha=0.01, device='cuda')
        
    except Exception as e:
        print(f"   [Error] GPU计算失败: {e}")
        print("   切换到 CPU (可能会很慢)...")
        F_batch = compute_mvgc_batch(segments, p=5, alpha=0.01, device='cpu')
        
    t1 = time.time()
    print(f"   计算完成! 耗时: {t1-t0:.4f} 秒")
    print(f"   输出形状: {F_batch.shape} (Batch, Channel, Channel)")
    
    # 4. 特征扁平化 (模仿你们之前的 pack_mvgc_from_matrix)
    # 将 (128, 128) 展平成特定的特征向量
    # 例如：保留每个通道的 Out-degree (列之和) 或 In-degree
    print("4. 结果后处理示例...")
    
    # 示例：计算每个片段的平均因果强度
    avg_causality = F_batch.mean(axis=(1, 2))
    for i, val in enumerate(avg_causality):
        print(f"   片段 {i}: 平均 GC 强度 = {val:.6f}")
        
    # 保存结果
    out_file = "demo_result_128.npz"
    np.savez(out_file, mvgc=F_batch, raw_data=segments)
    print(f"\n结果已保存至 {out_file}")

if __name__ == "__main__":
    demo_high_dim()