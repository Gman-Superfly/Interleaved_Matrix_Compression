import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .model import CompressedMLP
from .infer_cpu import forward_progressive_bucketing


@torch.no_grad()
def compute_hidden_strengths_percentiles(
    model: CompressedMLP,
    data_loader: DataLoader,
    interleaved_params: torch.Tensor,
    percentiles: Tuple[float, float] = (0.7, 0.85),
    max_batches: int = 50,
) -> Dict[str, torch.Tensor]:
    """Estimate per-layer thresholds for hidden-layer activation strengths using percentiles.

    Returns:
        dict with tensors of shape (num_layers_total,) including input layer, e.g.:
        {
          "hidden_thr2": Tensor[num_layers_total],
          "hidden_thr3": Tensor[num_layers_total],
        }
    """
    device = interleaved_params.device
    model.eval()
    layers_total = 1 + len(model.hidden_layers)  # input + hidden
    thr2_vals = [[] for _ in range(layers_total)]
    thr3_vals = [[] for _ in range(layers_total)]

    # Walk through interleaved params extracting w1/b1 per layer for band-1 preacts
    for batch_idx, (data, _) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        x = data.to(device).view(data.size(0), -1)
        idx = 0
        # Input layer
        in_w_shape = model.input_layer.weight.shape
        in_b_size = model.hidden_dim
        chunk_w = in_w_shape[0] * in_w_shape[1]
        w1 = interleaved_params[idx:idx+chunk_w].view(in_w_shape); idx += chunk_w
        idx += chunk_w  # w2
        idx += chunk_w  # w3
        b1 = interleaved_params[idx:idx+in_b_size]; idx += in_b_size
        idx += in_b_size  # b2
        idx += in_b_size  # b3
        z = (x @ w1.T) + b1
        strength = CompressedMLP.activation_strength(z)
        thr2_vals[0].append(strength)
        thr3_vals[0].append(strength)

        # Hidden layers (use band-1 preactivations)
        chunk_w = model.hidden_dim * model.hidden_dim
        bias_size = model.hidden_dim
        x_act = torch.relu(z)
        for li in range(len(model.hidden_layers)):
            w1 = interleaved_params[idx:idx+chunk_w].view(model.hidden_dim, model.hidden_dim); idx += chunk_w
            idx += chunk_w  # w2
            idx += chunk_w  # w3
            b1 = interleaved_params[idx:idx+bias_size]; idx += bias_size
            idx += bias_size  # b2
            idx += bias_size  # b3
            z = (x_act @ w1.T) + b1
            strength = CompressedMLP.activation_strength(z)
            thr2_vals[li + 1].append(strength)
            thr3_vals[li + 1].append(strength)
            x_act = torch.relu(z)

    # Concatenate collected strengths and compute percentiles
    thr2_t = []
    thr3_t = []
    p2, p3 = percentiles
    for li in range(layers_total):
        if len(thr2_vals[li]) == 0:
            thr2_t.append(torch.tensor(0.0, device=device))
            thr3_t.append(torch.tensor(0.0, device=device))
            continue
        s2 = torch.cat(thr2_vals[li], dim=0)
        s3 = torch.cat(thr3_vals[li], dim=0)
        thr2_t.append(torch.quantile(s2, q=p2))
        thr3_t.append(torch.quantile(s3, q=p3))
    return {
        "hidden_thr2": torch.stack(thr2_t),
        "hidden_thr3": torch.stack(thr3_t),
    }


@torch.no_grad()
def calibrate_output_threshold(
    model: CompressedMLP,
    data_loader: DataLoader,
    interleaved_params: torch.Tensor,
    target_expand_rate: float = 0.1,
    metric: str = "entropy",
    candidate_thresholds: Optional[torch.Tensor] = None,
    max_samples: int = 3000,
) -> float:
    """Calibrate final-layer uncertainty threshold to hit a target expansion rate."""
    model.eval()
    device = interleaved_params.device
    if candidate_thresholds is None:
        if metric == "entropy":
            candidate_thresholds = torch.linspace(0.2, 1.8, steps=9, device=device)
        else:
            candidate_thresholds = torch.linspace(0.2, 0.9, steps=8, device=device)
    seen = 0
    # Use band-1-only output uncertainty for decision, matching progressive gating logic
    rates = []
    for thr in candidate_thresholds:
        expanded = 0
        total = 0
        seen = 0
        for data, _ in data_loader:
            x = data.to(device)
            y1 = model.forward_banded(x, interleaved_params, bands=1)
            unc = model.compute_per_sample_uncertainty_from_logits(y1, metric=metric)
            expanded += (unc > float(thr.item())).sum().item()
            total += unc.numel()
            seen += x.size(0)
            if seen >= max_samples:
                break
        rate = expanded / max(1, total)
        rates.append((float(thr.item()), rate))
    best_thr = min(rates, key=lambda t: abs(t[1] - target_expand_rate))[0]
    return float(best_thr)


def save_thresholds_json(
    path: str,
    output_threshold: float,
    hidden_thresholds: Dict[str, torch.Tensor],
) -> None:
    out = {
        "output_threshold": float(output_threshold),
        "hidden_thr2": [float(x.item()) for x in hidden_thresholds["hidden_thr2"]],
        "hidden_thr3": [float(x.item()) for x in hidden_thresholds["hidden_thr3"]],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))


def load_thresholds_json(path: str) -> Dict[str, torch.Tensor]:
    p = Path(path)
    data = json.loads(p.read_text())
    return {
        "output_threshold": torch.tensor(data["output_threshold"], dtype=torch.float32),
        "hidden_thr2": torch.tensor(data["hidden_thr2"], dtype=torch.float32),
        "hidden_thr3": torch.tensor(data["hidden_thr3"], dtype=torch.float32),
    }


