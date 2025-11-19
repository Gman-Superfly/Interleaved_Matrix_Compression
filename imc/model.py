import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .compress import MatrixCompressor


class CompressedMLP(nn.Module):
    """MLP with lossless compression utilities and interleaved parameter vector helpers.

    Default config is intentionally large; prefer overriding in demos/tests.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, output_dim: int = 10, num_hidden_layers: int = 380):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.compressor = MatrixCompressor()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            self.input_layer.weight.copy_(self.compressor.warmup(self.input_layer.weight))
            self.input_layer.bias.copy_(self.compressor.warmup(self.input_layer.bias))
            for layer in self.hidden_layers:
                layer.weight.copy_(self.compressor.warmup(layer.weight))
                layer.bias.copy_(self.compressor.warmup(layer.bias))
            self.output_layer.weight.copy_(self.compressor.warmup(self.output_layer.weight))
            self.output_layer.bias.copy_(self.compressor.warmup(self.output_layer.bias))

    @staticmethod
    def compute_per_sample_uncertainty_from_logits(logits: torch.Tensor, metric: str = "entropy") -> torch.Tensor:
        assert logits is not None, "logits required"
        assert isinstance(logits, torch.Tensor), f"Expected torch.Tensor, got {type(logits)}"
        probs = torch.softmax(logits, dim=1)
        if metric == "entropy":
            return (-torch.sum(probs * torch.log(probs + 1e-10), dim=1))
        elif metric == "one_minus_max_prob":
            return (1.0 - probs.max(dim=1).values)
        else:
            raise ValueError(f"Unknown uncertainty metric: {metric}")

    @staticmethod
    def default_uncertainty_threshold(metric: str = "entropy") -> float:
        if metric == "entropy":
            return 1.0
        if metric == "one_minus_max_prob":
            return 0.5
        raise ValueError(f"Unknown uncertainty metric: {metric}")

    @staticmethod
    def activation_strength(activations: torch.Tensor) -> torch.Tensor:
        assert isinstance(activations, torch.Tensor), f"Expected torch.Tensor, got {type(activations)}"
        return activations.abs().mean(dim=1)

    def expected_interleaved_numel(self) -> int:
        total = 0
        # Input layer
        in_w = self.hidden_dim * self.input_layer.in_features
        in_b = self.hidden_dim
        total += 3 * in_w + 3 * in_b
        # Hidden layers
        hid_w = self.hidden_dim * self.hidden_dim
        hid_b = self.hidden_dim
        total += len(self.hidden_layers) * (3 * hid_w + 3 * hid_b)
        # Output layer
        out_w = self.output_layer.in_features * self.output_layer.out_features
        out_b = self.output_layer.out_features
        total += 3 * out_w + 3 * out_b
        return total

    def _validate_interleaved_vector(self, interleaved_params: torch.Tensor) -> None:
        assert isinstance(interleaved_params, torch.Tensor), "interleaved_params must be a Tensor"
        assert interleaved_params.dim() == 1, "interleaved_params must be a 1D tensor"
        expected = self.expected_interleaved_numel()
        assert interleaved_params.numel() == expected, f"interleaved_params length {interleaved_params.numel()} != expected {expected}"

    def forward_banded(self, x: torch.Tensor, interleaved_params: torch.Tensor, bands: int = 1) -> torch.Tensor:
        """Forward using only the first N bands per layer (N in {1,2,3})."""
        self._validate_interleaved_vector(interleaved_params)
        assert bands in (1, 2, 3), f"bands must be 1, 2 or 3, got {bands}"
        x = x.view(x.size(0), -1)
        idx = 0
        # Input layer
        chunk_size = self.hidden_dim * self.input_layer.in_features
        w1 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape); idx += chunk_size
        w2 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape); idx += chunk_size
        w3 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape); idx += chunk_size
        chunk_size = self.hidden_dim
        b1 = interleaved_params[idx:idx+chunk_size]; idx += chunk_size
        b2 = interleaved_params[idx:idx+chunk_size]; idx += chunk_size
        b3 = interleaved_params[idx:idx+chunk_size]; idx += chunk_size
        z = torch.matmul(x, w1.T) + b1
        if bands >= 2:
            z = z + torch.matmul(x, w2.T) + b2
        if bands >= 3:
            z = z + torch.matmul(x, w3.T) + b3
        x = torch.relu(z)
        # Hidden layers
        chunk_size_w = self.hidden_dim * self.hidden_dim
        bias_size = self.hidden_dim
        for _ in range(len(self.hidden_layers)):
            w1 = interleaved_params[idx:idx+chunk_size_w].view(self.hidden_dim, self.hidden_dim); idx += chunk_size_w
            w2 = interleaved_params[idx:idx+chunk_size_w].view(self.hidden_dim, self.hidden_dim); idx += chunk_size_w
            w3 = interleaved_params[idx:idx+chunk_size_w].view(self.hidden_dim, self.hidden_dim); idx += chunk_size_w
            b1 = interleaved_params[idx:idx+bias_size]; idx += bias_size
            b2 = interleaved_params[idx:idx+bias_size]; idx += bias_size
            b3 = interleaved_params[idx:idx+bias_size]; idx += bias_size
            z = torch.matmul(x, w1.T) + b1
            if bands >= 2:
                z = z + torch.matmul(x, w2.T) + b2
            if bands >= 3:
                z = z + torch.matmul(x, w3.T) + b3
            x = torch.relu(z)
        # Output layer
        chunk_size = self.output_layer.in_features * self.output_layer.out_features
        bias_size = self.output_layer.out_features
        w1 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape); idx += chunk_size
        w2 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape); idx += chunk_size
        w3 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape); idx += chunk_size
        b1 = interleaved_params[idx:idx+bias_size]; idx += bias_size
        b2 = interleaved_params[idx:idx+bias_size]; idx += bias_size
        b3 = interleaved_params[idx:idx+bias_size]
        y = torch.matmul(x, w1.T) + b1
        if bands >= 2:
            y = y + torch.matmul(x, w2.T) + b2
        if bands >= 3:
            y = y + torch.matmul(x, w3.T) + b3
        # idx should equal total length
        assert idx + bias_size == interleaved_params.numel(), "forward_banded index mismatch"
        return y

    def interleave_params(self) -> torch.Tensor:
        params = []
        w1, w2, w3 = self.compressor.split(self.input_layer.weight)
        b1, b2, b3 = self.compressor.split(self.input_layer.bias)
        params.extend([w1.view(-1), w2.view(-1), w3.view(-1), b1, b2, b3])
        for layer in self.hidden_layers:
            w1, w2, w3 = self.compressor.split(layer.weight)
            b1, b2, b3 = self.compressor.split(layer.bias)
            params.extend([w1.view(-1), w2.view(-1), w3.view(-1), b1, b2, b3])
        w1, w2, w3 = self.compressor.split(self.output_layer.weight)
        b1, b2, b3 = self.compressor.split(self.output_layer.bias)
        params.extend([w1.view(-1), w2.view(-1), w3.view(-1), b1, b2, b3])
        return torch.cat(params)

    def compute_band3_l2(self, interleaved_params: torch.Tensor) -> torch.Tensor:
        """Compute L2 norm of all band-3 parameters (w3 and b3 across layers)."""
        self._validate_interleaved_vector(interleaved_params)
        idx = 0
        l2 = torch.zeros((), dtype=interleaved_params.dtype, device=interleaved_params.device)
        # Input layer
        chunk_size = self.hidden_dim * self.input_layer.in_features
        idx += chunk_size  # w1
        idx += chunk_size  # w2
        w3 = interleaved_params[idx:idx+chunk_size]; idx += chunk_size
        chunk_size_b = self.hidden_dim
        idx += chunk_size_b  # b1
        idx += chunk_size_b  # b2
        b3 = interleaved_params[idx:idx+chunk_size_b]; idx += chunk_size_b
        l2 = l2 + torch.sum(w3 * w3) + torch.sum(b3 * b3)
        # Hidden layers
        chunk_size_w = self.hidden_dim * self.hidden_dim
        bias_size = self.hidden_dim
        for _ in range(len(self.hidden_layers)):
            idx += chunk_size_w  # w1
            idx += chunk_size_w  # w2
            w3 = interleaved_params[idx:idx+chunk_size_w]; idx += chunk_size_w
            idx += bias_size  # b1
            idx += bias_size  # b2
            b3 = interleaved_params[idx:idx+bias_size]; idx += bias_size
            l2 = l2 + torch.sum(w3 * w3) + torch.sum(b3 * b3)
        # Output layer
        chunk_size = self.output_layer.in_features * self.output_layer.out_features
        bias_size = self.output_layer.out_features
        idx += chunk_size  # w1
        idx += chunk_size  # w2
        w3 = interleaved_params[idx:idx+chunk_size]; idx += chunk_size
        idx += bias_size  # b1
        idx += bias_size  # b2
        b3 = interleaved_params[idx:idx+bias_size]
        l2 = l2 + torch.sum(w3 * w3) + torch.sum(b3 * b3)
        # idx should equal total length
        assert idx + bias_size == interleaved_params.numel(), "compute_band3_l2 index mismatch"
        return l2

    def deinterleave_params(self, interleaved: torch.Tensor) -> None:
        self._validate_interleaved_vector(interleaved)
        idx = 0
        with torch.no_grad():
            chunk_size = self.hidden_dim * self.input_layer.in_features
            w1 = interleaved[idx:idx+chunk_size].view(self.input_layer.weight.shape)
            idx += chunk_size
            w2 = interleaved[idx:idx+chunk_size].view(self.input_layer.weight.shape)
            idx += chunk_size
            w3 = interleaved[idx:idx+chunk_size].view(self.input_layer.weight.shape)
            idx += chunk_size
            self.input_layer.weight.copy_(self.compressor.merge(w1, w2, w3))
            chunk_size = self.hidden_dim
            b1 = interleaved[idx:idx+chunk_size]
            idx += chunk_size
            b2 = interleaved[idx:idx+chunk_size]
            idx += chunk_size
            b3 = interleaved[idx:idx+chunk_size]
            idx += chunk_size
            self.input_layer.bias.copy_(self.compressor.merge(b1, b2, b3))
            chunk_size = self.hidden_dim * self.hidden_dim
            bias_size = self.hidden_dim
            for layer in self.hidden_layers:
                w1 = interleaved[idx:idx+chunk_size].view(layer.weight.shape)
                idx += chunk_size
                w2 = interleaved[idx:idx+chunk_size].view(layer.weight.shape)
                idx += chunk_size
                w3 = interleaved[idx:idx+chunk_size].view(layer.weight.shape)
                idx += chunk_size
                layer.weight.copy_(self.compressor.merge(w1, w2, w3))
                b1 = interleaved[idx:idx+bias_size]
                idx += bias_size
                b2 = interleaved[idx:idx+bias_size]
                idx += bias_size
                b3 = interleaved[idx:idx+bias_size]
                idx += bias_size
                layer.bias.copy_(self.compressor.merge(b1, b2, b3))
            chunk_size = self.output_layer.in_features * self.output_layer.out_features
            bias_size = self.output_layer.out_features
            w1 = interleaved[idx:idx+chunk_size].view(self.output_layer.weight.shape)
            idx += chunk_size
            w2 = interleaved[idx:idx+chunk_size].view(self.output_layer.weight.shape)
            idx += chunk_size
            w3 = interleaved[idx:idx+chunk_size].view(self.output_layer.weight.shape)
            idx += chunk_size
            self.output_layer.weight.copy_(self.compressor.merge(w1, w2, w3))
            b1 = interleaved[idx:idx+bias_size]
            idx += bias_size
            b2 = interleaved[idx:idx+bias_size]
            idx += bias_size
            b3 = interleaved[idx:idx+bias_size]
            self.output_layer.bias.copy_(self.compressor.merge(b1, b2, b3))
            # idx should equal total length
            assert idx + bias_size == interleaved.numel(), "deinterleave_params index mismatch"

    def forward_expanded(self, x: torch.Tensor, interleaved_params: torch.Tensor) -> torch.Tensor:
        self._validate_interleaved_vector(interleaved_params)
        x = x.view(x.size(0), -1)
        idx = 0
        # Input layer
        chunk_size = self.hidden_dim * self.input_layer.in_features
        w1 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape); idx += chunk_size
        w2 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape); idx += chunk_size
        w3 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape); idx += chunk_size
        chunk_size = self.hidden_dim
        b1 = interleaved_params[idx:idx+chunk_size]; idx += chunk_size
        b2 = interleaved_params[idx:idx+chunk_size]; idx += chunk_size
        b3 = interleaved_params[idx:idx+chunk_size]; idx += chunk_size
        x = torch.relu(torch.matmul(x, w1.T) + b1 + torch.matmul(x, w2.T) + b2 + torch.matmul(x, w3.T) + b3)
        # Hidden layers
        chunk_size = self.hidden_dim * self.hidden_dim
        bias_size = self.hidden_dim
        for _ in range(len(self.hidden_layers)):
            w1 = interleaved_params[idx:idx+chunk_size].view(self.hidden_dim, self.hidden_dim); idx += chunk_size
            w2 = interleaved_params[idx:idx+chunk_size].view(self.hidden_dim, self.hidden_dim); idx += chunk_size
            w3 = interleaved_params[idx:idx+chunk_size].view(self.hidden_dim, self.hidden_dim); idx += chunk_size
            b1 = interleaved_params[idx:idx+bias_size]; idx += bias_size
            b2 = interleaved_params[idx:idx+bias_size]; idx += bias_size
            b3 = interleaved_params[idx:idx+bias_size]; idx += bias_size
            x = torch.relu(torch.matmul(x, w1.T) + b1 + torch.matmul(x, w2.T) + b2 + torch.matmul(x, w3.T) + b3)
        # Output layer
        chunk_size = self.output_layer.in_features * self.output_layer.out_features
        bias_size = self.output_layer.out_features
        w1 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape); idx += chunk_size
        w2 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape); idx += chunk_size
        w3 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape); idx += chunk_size
        b1 = interleaved_params[idx:idx+bias_size]; idx += bias_size
        b2 = interleaved_params[idx:idx+bias_size]; idx += bias_size
        b3 = interleaved_params[idx:idx+bias_size]
        x = torch.matmul(x, w1.T) + b1 + torch.matmul(x, w2.T) + b2 + torch.matmul(x, w3.T) + b3
        # idx should equal total length
        assert idx + bias_size == interleaved_params.numel(), "forward_expanded index mismatch"
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merged path using module weights (baseline)."""
        x = x.view(x.size(0), -1)
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)


