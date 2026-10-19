# PyTorch-MVGC

[![PyPI version](https://badge.fury.io/py/pytorch-mvgc.svg)](https://badge.fury.io/py/pytorch-mvgc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PyTorch-MVGC** is a GPU-accelerated Python implementation of **Multivariate Granger Causality (MVGC)** analysis. 

It is designed for high-dimensional time-series data (e.g., EEG, fMRI, MEA) where traditional CPU-based implementations are too slow. By leveraging PyTorch tensors and batched matrix operations, this library achieves significant speedups compared to standard NumPy or Matlab implementations.

## Features

- **🚀 GPU Acceleration**: Fully implemented in PyTorch, supporting CUDA acceleration out-of-the-box.
- **⚡ Batch Processing**: Optimized for processing multiple trials or sliding windows simultaneously.
- **🔧 Numerical Stability**: Implements Ridge regularization and Pseudo-inverse solvers to handle ill-conditioned matrices common in high-channel neural recordings (e.g., 128-channel EEG).
- **🐍 Python Native**: NumPy inputs and outputs for Python time-series pipelines.

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


## API and compatibility

Python 3.9 or newer is required. Inputs are real, finite numeric arrays (or
array-like sequences); results are NumPy float64 arrays. Both import names are
supported: `pytorch_mvgc` is the documented name and `fast_mvgc` remains compatible
with 0.1.0 code.

- `compute_mvgc(X, p, alpha=0.01, device="cuda")`: input `(channels, time)`,
  output `(target, source)`.
- `compute_mvgc_batch(X, p, alpha=0.01, device="cuda")`: input
  `(batch, channels, time)`, output `(batch, target, source)`.
- `compute_mvgc_cpu(X, p, alpha=0.01)`: NumPy reference engine with the same
  residual sum-of-squares convention as the PyTorch engine.
- `segment_data(X, window_size, step_size=None)`: returns complete windows as
  `(windows, channels, window_size)`. The default stride is `window_size`;
  incomplete trailing windows are discarded.

`p` must be a positive integer smaller than the time length. `alpha` must be
finite and nonnegative; zero selects unregularized least squares and supports
rank-deficient designs. Regularized regression remains the default. Segmentation
sizes must be positive integers and at least one complete window must fit.

The implementation fits full and source-excluded finite-order regressions and
returns nonnegative log residual ratios, with a zero diagonal. This is not an
implementation of the MATLAB toolbox's autocovariance/state-space pipeline.

CUDA requests (including `cuda:0` and `torch.device("cuda:0")`) fall back to CPU
with a warning when CUDA is unavailable. A CUDA-enabled PyTorch build compatible
with your GPU is required for acceleration. For NVIDIA Blackwell GPUs such as
RTX 50-series cards, use a PyTorch build with Blackwell support (introduced in
PyTorch 2.7 with CUDA 12.8); see the [official installation selector](https://pytorch.org/get-started/locally/)
and [PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/).
Select `device="cpu"` to explicitly run on CPU.

```python
from pytorch_mvgc import compute_mvgc_batch, segment_data

windows = segment_data(X, window_size=200, step_size=100)
F_batch = compute_mvgc_batch(windows, p=2, device="cpu")
```

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m pip install build twine
python -m build
python -m twine check dist/*
```

GPU parity tests require a compatible CUDA build and GPU. They are skipped when
CUDA is unavailable or an older build lacks support for Blackwell hardware.

## 0.1.1 fixes

- Ship the documented `pytorch_mvgc` namespace while retaining `fast_mvgc`.
- Remove the correct source-channel columns at every lag for `p > 1`.
- Handle singular unregularized regressions with least squares / pseudoinverse.
- Align NumPy and PyTorch residual ratios and validate arrays, lag order,
  regularization, and window sizes before computation.
- Support list inputs, negative-stride arrays, and indexed CUDA CPU fallback.
- Correct project metadata and UTF-8 README loading for package builds.
