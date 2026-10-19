import os

# Avoid duplicate OpenMP runtimes in mixed Anaconda NumPy / pip PyTorch installs.
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
