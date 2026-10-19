"""Legacy import namespace, supported alongside pytorch_mvgc."""
from .core import compute_mvgc, compute_mvgc_batch
from .cpu_engine import compute_mvgc_cpu
from .utils import segment_data

__version__ = "0.1.1"
__all__ = ["compute_mvgc", "compute_mvgc_batch", "compute_mvgc_cpu", "segment_data"]
