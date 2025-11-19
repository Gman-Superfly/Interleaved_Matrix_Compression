from typing import Dict, Optional, Tuple, Union
import torch

from .model import CompressedMLP
from .infer_cpu import forward_progressive_bucketing as _cpu_forward_progressive_bucketing


def forward_progressive_bucketing(
    model: CompressedMLP,
    x: torch.Tensor,
    interleaved_params: torch.Tensor,
    act_threshold2: float = 0.02,
    act_threshold3: float = 0.01,
    final_metric: str = "entropy",
    final_threshold: Optional[float] = None,
    return_stats: bool = False,
    per_layer_thresholds: Optional[Dict[str, torch.Tensor]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """GPU-ready progressive bucketing.

    Current implementation mirrors the CPU bucketing logic and runs efficiently on CUDA tensors.
    Future work: specialize with fused/scatter kernels (e.g., Triton) and block-level routing.
    """
    return _cpu_forward_progressive_bucketing(
        model=model,
        x=x,
        interleaved_params=interleaved_params,
        act_threshold2=act_threshold2,
        act_threshold3=act_threshold3,
        final_metric=final_metric,
        final_threshold=final_threshold,
        return_stats=return_stats,
        per_layer_thresholds=per_layer_thresholds,
    )


