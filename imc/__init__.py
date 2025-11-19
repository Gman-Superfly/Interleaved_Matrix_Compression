"""Interleaved Matrix Compression (IMC) package.

Public API:
- MatrixCompressor: lossless banded split/merge utilities
- CompressedMLP: model with interleaved parameter layout helpers
- forward_progressive_bucketing: compute-efficient progressive inference on CPU
"""

from .compress import MatrixCompressor
from .model import CompressedMLP
from .infer_cpu import forward_progressive_bucketing

__all__ = [
    "MatrixCompressor",
    "CompressedMLP",
    "forward_progressive_bucketing",
]


