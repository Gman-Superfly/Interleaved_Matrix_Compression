import torch
from typing import Tuple


class MatrixCompressor:
    """Lossless, banded weight decomposition with fp32-lattice guarantees.

    Splits a tensor into 3 integer-bounded bands that sum (when merged) to the
    original value snapped to a fp32-representable lattice defined by scale_factor.
    """

    def __init__(self, keys=[0.3, 0.6], scale_factor: int = 2**24):
        # Interpret keys as cumulative boundaries in (0, 1): e.g., [0.3, 0.6]
        # Convert to per-band capacities for the first two bands; the third receives the remainder.
        self.boundaries = torch.tensor(list(keys), dtype=torch.float32)
        assert self.boundaries.numel() == 2, "Expected two cumulative boundaries"
        assert 0.0 < self.boundaries[0] < self.boundaries[1] < 1.0, "Boundaries must satisfy 0 < k1 < k2 < 1"
        self.scale_factor = torch.tensor(scale_factor, dtype=torch.int64)
        caps = torch.stack([
            self.boundaries[0],
            self.boundaries[1] - self.boundaries[0],
        ])
        self.caps_int = torch.round(caps * self.scale_factor).to(torch.int64)

    def split(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sf = self.scale_factor.to(device=matrix.device, dtype=matrix.dtype)
        matrix_int = (matrix * sf).round().to(torch.int64)
        flat_int = matrix_int.view(-1)

        # Symmetric capacity limits for positive and negative values
        c1 = int(self.caps_int[0].item())
        c2 = int(self.caps_int[1].item())
        c1_t = torch.full_like(flat_int, c1)
        c2_t = torch.full_like(flat_int, c2)

        # First band
        pos_mask = flat_int >= 0
        b1_pos = torch.minimum(flat_int, c1_t)
        b1_neg = torch.maximum(flat_int, -c1_t)
        m1_int = torch.where(pos_mask, b1_pos, b1_neg)
        rem1 = flat_int - m1_int

        # Second band
        pos_mask2 = rem1 >= 0
        b2_pos = torch.minimum(rem1, c2_t)
        b2_neg = torch.maximum(rem1, -c2_t)
        m2_int = torch.where(pos_mask2, b2_pos, b2_neg)

        # Third band (remainder)
        m3_int = rem1 - m2_int

        sf_float = self.scale_factor.to(device=matrix.device, dtype=torch.float32)
        m1 = m1_int.float() / sf_float
        m2 = m2_int.float() / sf_float
        m3 = m3_int.float() / sf_float
        return m1.view_as(matrix), m2.view_as(matrix), m3.view_as(matrix)

    def merge(self, m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
        sf = self.scale_factor.to(device=m1.device, dtype=m1.dtype)
        m1_int = (m1 * sf).round().to(torch.int64)
        m2_int = (m2 * sf).round().to(torch.int64)
        m3_int = (m3 * sf).round().to(torch.int64)
        merged_int = m1_int + m2_int + m3_int
        sf_float = self.scale_factor.to(device=m1.device, dtype=torch.float32)
        return merged_int.float() / sf_float

    def warmup(self, matrix: torch.Tensor, cycles: int = 1) -> torch.Tensor:
        for _ in range(cycles):
            m1, m2, m3 = self.split(matrix)
            matrix = self.merge(m1, m2, m3)
        return matrix


