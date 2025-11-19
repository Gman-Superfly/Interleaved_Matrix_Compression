import torch
from typing import Dict, Optional, Tuple, Union

from .model import CompressedMLP


def _slice_layer_params(
    interleaved_params: torch.Tensor,
    idx: int,
    weight_shape,
    bias_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return (w1, w2, w3, b1, b2, b3, new_idx) and updated index."""
    chunk_w = weight_shape[0] * weight_shape[1]
    w1 = interleaved_params[idx:idx+chunk_w].view(weight_shape); idx += chunk_w
    w2 = interleaved_params[idx:idx+chunk_w].view(weight_shape); idx += chunk_w
    w3 = interleaved_params[idx:idx+chunk_w].view(weight_shape); idx += chunk_w
    b1 = interleaved_params[idx:idx+bias_size]; idx += bias_size
    b2 = interleaved_params[idx:idx+bias_size]; idx += bias_size
    b3 = interleaved_params[idx:idx+bias_size]; idx += bias_size
    return w1, w2, w3, b1, b2, b3, idx


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
    """Compute-efficient, single-pass progressive unfolding on CPU using per-sample bucketing.

    - Always compute band-1.
    - Compute band-2 only for samples that request it (idx2).
    - Compute band-3 only for samples that request it (idx3).
    - Output layer uses uncertainty-based gating similarly.
    - Optionally use per-layer thresholds from calibration.
    """
    model._validate_interleaved_vector(interleaved_params)
    x = x.view(x.size(0), -1)
    B = x.size(0)
    idx = 0

    # Input layer
    in_w_shape = model.input_layer.weight.shape
    in_b_size = model.hidden_dim
    w1, w2, w3, b1, b2, b3, idx = _slice_layer_params(interleaved_params, idx, in_w_shape, in_b_size)
    z = torch.matmul(x, w1.T) + b1
    strength1 = model.activation_strength(z)
    thr2 = act_threshold2
    thr3 = act_threshold3
    if per_layer_thresholds is not None and "hidden_thr2" in per_layer_thresholds and "hidden_thr3" in per_layer_thresholds:
        # Layer 0 thresholds if provided
        thr2 = float(per_layer_thresholds["hidden_thr2"][0].item())
        thr3 = float(per_layer_thresholds["hidden_thr3"][0].item())
    idx2 = (strength1 < thr2).nonzero(as_tuple=False).squeeze(1)
    if idx2.numel() > 0:
        z[idx2] = z[idx2] + (x[idx2] @ w2.T) + b2
    strength2 = model.activation_strength(z)
    idx3 = (strength2 < thr3).nonzero(as_tuple=False).squeeze(1)
    if idx3.numel() > 0:
        z[idx3] = z[idx3] + (x[idx3] @ w3.T) + b3
    x_act = torch.relu(z)

    if return_stats:
        hidden_masks2_total = float(idx2.numel())
        hidden_masks3_total = float(idx3.numel())
        hidden_masks_den = float(B)

    # Hidden layers
    chunk_size_w = model.hidden_dim * model.hidden_dim
    bias_size = model.hidden_dim
    for layer_idx in range(len(model.hidden_layers)):
        w1 = interleaved_params[idx:idx+chunk_size_w].view(model.hidden_dim, model.hidden_dim); idx += chunk_size_w
        w2 = interleaved_params[idx:idx+chunk_size_w].view(model.hidden_dim, model.hidden_dim); idx += chunk_size_w
        w3 = interleaved_params[idx:idx+chunk_size_w].view(model.hidden_dim, model.hidden_dim); idx += chunk_size_w
        b1 = interleaved_params[idx:idx+bias_size]; idx += bias_size
        b2 = interleaved_params[idx:idx+bias_size]; idx += bias_size
        b3 = interleaved_params[idx:idx+bias_size]; idx += bias_size

        z = (x_act @ w1.T) + b1
        strength1 = model.activation_strength(z)
        thr2 = act_threshold2
        thr3 = act_threshold3
        if per_layer_thresholds is not None and "hidden_thr2" in per_layer_thresholds and "hidden_thr3" in per_layer_thresholds:
            thr2 = float(per_layer_thresholds["hidden_thr2"][layer_idx + 1].item())
            thr3 = float(per_layer_thresholds["hidden_thr3"][layer_idx + 1].item())
        idx2 = (strength1 < thr2).nonzero(as_tuple=False).squeeze(1)
        if idx2.numel() > 0:
            z[idx2] = z[idx2] + (x_act[idx2] @ w2.T) + b2
        strength2 = model.activation_strength(z)
        idx3 = (strength2 < thr3).nonzero(as_tuple=False).squeeze(1)
        if idx3.numel() > 0:
            z[idx3] = z[idx3] + (x_act[idx3] @ w3.T) + b3
        x_act = torch.relu(z)
        if return_stats:
            hidden_masks2_total += float(idx2.numel())
            hidden_masks3_total += float(idx3.numel())
            hidden_masks_den += float(B)

    # Output layer
    out_w_shape = model.output_layer.weight.shape
    out_b_size = model.output_layer.out_features
    w1, w2, w3, b1, b2, b3, idx = _slice_layer_params(interleaved_params, idx, out_w_shape, out_b_size)

    y1 = (x_act @ w1.T) + b1
    if final_threshold is None:
        final_threshold = model.default_uncertainty_threshold(final_metric)
    unc1 = model.compute_per_sample_uncertainty_from_logits(y1, metric=final_metric)
    idx2_out = (unc1 > final_threshold).nonzero(as_tuple=False).squeeze(1)
    y = y1.clone()
    if idx2_out.numel() > 0:
        y[idx2_out] = y[idx2_out] + (x_act[idx2_out] @ w2.T) + b2
    unc2 = model.compute_per_sample_uncertainty_from_logits(y, metric=final_metric)
    idx3_out = (unc2 > final_threshold).nonzero(as_tuple=False).squeeze(1)
    if idx3_out.numel() > 0:
        y[idx3_out] = y[idx3_out] + (x_act[idx3_out] @ w3.T) + b3

    # idx should equal total length
    assert idx + out_b_size == interleaved_params.numel(), "forward_progressive_bucketing index mismatch"

    if return_stats:
        stats = {
            "out_expand2_rate": float(idx2_out.numel()) / float(B),
            "out_expand3_rate": float(idx3_out.numel()) / float(B),
            "hidden_expand2_rate": hidden_masks2_total / hidden_masks_den,
            "hidden_expand3_rate": hidden_masks3_total / hidden_masks_den,
        }
        return y, stats
    return y


