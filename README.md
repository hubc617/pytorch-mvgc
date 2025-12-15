# PyTorch-MVGC

[![PyPI version](https://badge.fury.io/py/pytorch-mvgc.svg)](https://badge.fury.io/py/pytorch-mvgc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PyTorch-MVGC** is a GPU-accelerated Python implementation of **Multivariate Granger Causality (MVGC)** analysis. 

It is designed for high-dimensional time-series data (e.g., EEG, fMRI, MEA) where traditional CPU-based implementations are too slow. By leveraging PyTorch tensors and batched matrix operations, this library achieves significant speedups compared to standard NumPy or Matlab implementations.

## Features

- **🚀 GPU Acceleration**: Fully implemented in PyTorch, supporting CUDA acceleration out-of-the-box.
- **⚡ Batch Processing**: Optimized for processing multiple trials or sliding windows simultaneously.
- **🔧 Numerical Stability**: Implements Ridge regularization and Pseudo-inverse solvers to handle ill-conditioned matrices common in high-channel neural recordings (e.g., 128-channel EEG).
- **🐍 Python Native**: A seamless replacement for the Matlab MVGC toolbox for Python pipelines.

## Installation

```bash
pip install pytorch-mvgc
```

## Quick Start

```python
import numpy as np
import torch
from pytorch_mvgc import compute_mvgc

# 1. Generate sample data (3 channels, 1000 timepoints)
# System: Node 0 drives Node 1
n_time = 1000
X = np.zeros((3, n_time))
noise = 0.5 * np.random.randn(3, n_time)
for t in range(1, n_time):
    X[0, t] = 0.8 * X[0, t-1] + noise[0, t]
    X[1, t] = 0.5 * X[1, t-1] + 0.5 * X[0, t-1] + noise[1, t] # 0->1 Causal
    X[2, t] = 0.7 * X[2, t-1] + noise[2, t]

# 2. Compute MVGC
# Returns matrix F where F[i, j] denotes causality j -> i
F_matrix = compute_mvgc(X, p=1, device='cuda')

print("MVGC Matrix:\n", np.round(F_matrix, 4))
# Expected: F[1, 0] should be significant (> 0)
```

