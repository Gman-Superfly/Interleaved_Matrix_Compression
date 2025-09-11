import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Lossless Matrix Compressor
class MatrixCompressor:
    def __init__(self, keys=[0.3, 0.6], scale_factor=2**24):
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
        
    def split(self, matrix):
        matrix_int = (matrix * self.scale_factor).round().to(torch.int64)
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

        m1 = m1_int.float() / self.scale_factor
        m2 = m2_int.float() / self.scale_factor
        m3 = m3_int.float() / self.scale_factor
        return m1.view_as(matrix), m2.view_as(matrix), m3.view_as(matrix)
    
    def merge(self, m1, m2, m3):
        m1_int = (m1 * self.scale_factor).round().to(torch.int64)
        m2_int = (m2 * self.scale_factor).round().to(torch.int64)
        m3_int = (m3 * self.scale_factor).round().to(torch.int64)
        merged_int = m1_int + m2_int + m3_int
        return merged_int.float() / self.scale_factor
    
    def warmup(self, matrix, cycles=1):
        for _ in range(cycles):
            m1, m2, m3 = self.split(matrix)
            matrix = self.merge(m1, m2, m3)
        return matrix

# MLP with lossless compression
class CompressedMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10, num_hidden_layers=380):
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
    def compute_uncertainty_from_logits(logits: torch.Tensor, metric: str = "entropy") -> float:
        """Compute batch uncertainty as a scalar.

        Args:
            logits: Network outputs of shape (batch_size, num_classes).
            metric: 'entropy' or 'one_minus_max_prob'.

        Returns:
            Float scalar uncertainty (Python float).
        """
        assert logits is not None, "logits required"
        assert isinstance(logits, torch.Tensor), f"Expected torch.Tensor, got {type(logits)}"
        probs = torch.softmax(logits, dim=1)
        if metric == "entropy":
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
            return float(entropy.item())
        elif metric == "one_minus_max_prob":
            one_minus_max = (1.0 - probs.max(dim=1).values).mean()
            return float(one_minus_max.item())
        else:
            raise ValueError(f"Unknown uncertainty metric: {metric}")

    @staticmethod
    def default_uncertainty_threshold(metric: str = "entropy") -> float:
        """Provide a sane default threshold per metric.

        For 10-way classification:
        - entropy in [0, ln(10)≈2.30]; default 1.0 is moderate uncertainty
        - one_minus_max_prob in [0, 0.9]; default 0.5 means max-prob < 0.5
        """
        if metric == "entropy":
            return 1.0
        if metric == "one_minus_max_prob":
            return 0.5
        raise ValueError(f"Unknown uncertainty metric: {metric}")

    @staticmethod
    def compute_per_sample_uncertainty_from_logits(logits: torch.Tensor, metric: str = "entropy") -> torch.Tensor:
        """Per-sample uncertainty vector.

        Args:
            logits: (batch_size, num_classes)
            metric: 'entropy' or 'one_minus_max_prob'

        Returns:
            Tensor of shape (batch_size,)
        """
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
    def activation_strength(activations: torch.Tensor) -> torch.Tensor:
        """Per-sample mean absolute activation strength."""
        assert isinstance(activations, torch.Tensor), f"Expected torch.Tensor, got {type(activations)}"
        return activations.abs().mean(dim=1)

    def forward_banded(self, x: torch.Tensor, interleaved_params: torch.Tensor, bands: int = 1) -> torch.Tensor:
        """Forward using only the first N bands per layer (N in {1,2,3})."""
        assert isinstance(interleaved_params, torch.Tensor), "interleaved_params must be a Tensor"
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
        return y

    def compute_band3_l2(self, interleaved_params: torch.Tensor) -> torch.Tensor:
        """Compute L2 norm of all band-3 parameters (w3 and b3 across layers)."""
        assert isinstance(interleaved_params, torch.Tensor), "interleaved_params must be a Tensor"
        idx = 0
        l2 = 0.0
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
        return l2

    def interleave_params(self):
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
    
    def deinterleave_params(self, interleaved):
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
    
    def forward_expanded(self, x, interleaved_params):
        """Forward pass using interleaved (split) weights."""
        x = x.view(x.size(0), -1)
        idx = 0
        # Input layer
        chunk_size = self.hidden_dim * self.input_layer.in_features
        w1 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape)
        idx += chunk_size
        w2 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape)
        idx += chunk_size
        w3 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape)
        idx += chunk_size
        chunk_size = self.hidden_dim
        b1 = interleaved_params[idx:idx+chunk_size]
        idx += chunk_size
        b2 = interleaved_params[idx:idx+chunk_size]
        idx += chunk_size
        b3 = interleaved_params[idx:idx+chunk_size]
        idx += chunk_size
        x = torch.relu(torch.matmul(x, w1.T) + b1 + torch.matmul(x, w2.T) + b2 + torch.matmul(x, w3.T) + b3)
        # Hidden layers
        chunk_size = self.hidden_dim * self.hidden_dim
        bias_size = self.hidden_dim
        for _ in range(len(self.hidden_layers)):
            w1 = interleaved_params[idx:idx+chunk_size].view(self.hidden_dim, self.hidden_dim)
            idx += chunk_size
            w2 = interleaved_params[idx:idx+chunk_size].view(self.hidden_dim, self.hidden_dim)
            idx += chunk_size
            w3 = interleaved_params[idx:idx+chunk_size].view(self.hidden_dim, self.hidden_dim)
            idx += chunk_size
            b1 = interleaved_params[idx:idx+bias_size]
            idx += bias_size
            b2 = interleaved_params[idx:idx+bias_size]
            idx += bias_size
            b3 = interleaved_params[idx:idx+bias_size]
            idx += bias_size
            x = torch.relu(torch.matmul(x, w1.T) + b1 + torch.matmul(x, w2.T) + b2 + torch.matmul(x, w3.T) + b3)
        # Output layer
        chunk_size = self.output_layer.in_features * self.output_layer.out_features
        bias_size = self.output_layer.out_features
        w1 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape)
        idx += chunk_size
        w2 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape)
        idx += chunk_size
        w3 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape)
        idx += chunk_size
        b1 = interleaved_params[idx:idx+bias_size]
        idx += bias_size
        b2 = interleaved_params[idx:idx+bias_size]
        idx += bias_size
        b3 = interleaved_params[idx:idx+bias_size]
        x = torch.matmul(x, w1.T) + b1 + torch.matmul(x, w2.T) + b2 + torch.matmul(x, w3.T) + b3
        return x
    
    def forward(self, x, uncertainty=None, threshold=0.5, interleaved_params=None):
        """Forward pass with dynamic unzip. Expand when uncertainty (e.g., entropy) is high."""
        if uncertainty is not None and uncertainty > threshold and interleaved_params is not None:
            return self.forward_expanded(x, interleaved_params)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

    def forward_progressive(
        self,
        x: torch.Tensor,
        interleaved_params: torch.Tensor,
        act_threshold2: float = 0.02,
        act_threshold3: float = 0.01,
        final_metric: str = "entropy",
        final_threshold: float = None,
    , return_stats: bool = False) -> torch.Tensor:
        """Single-pass progressive unfolding with per-layer gating.

        - Uses only band-1 by default, and conditionally adds band-2 and band-3 per sample.
        - Hidden layers use activation strength gating; output layer uses classification uncertainty.
        """
        assert isinstance(interleaved_params, torch.Tensor), "interleaved_params must be a Tensor"
        x = x.view(x.size(0), -1)
        idx = 0
        batch_size = x.size(0)

        # Input layer
        chunk_size = self.hidden_dim * self.input_layer.in_features
        w1 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape)
        idx += chunk_size
        w2 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape)
        idx += chunk_size
        w3 = interleaved_params[idx:idx+chunk_size].view(self.input_layer.weight.shape)
        idx += chunk_size
        chunk_size = self.hidden_dim
        b1 = interleaved_params[idx:idx+chunk_size]
        idx += chunk_size
        b2 = interleaved_params[idx:idx+chunk_size]
        idx += chunk_size
        b3 = interleaved_params[idx:idx+chunk_size]
        idx += chunk_size

        z1 = torch.matmul(x, w1.T) + b1
        strength1 = self.activation_strength(z1)
        mask2 = (strength1 < act_threshold2).unsqueeze(1).to(z1.dtype)
        z = z1 + (torch.matmul(x, w2.T) + b2) * mask2
        strength2 = self.activation_strength(z)
        mask3 = (strength2 < act_threshold3).unsqueeze(1).to(z1.dtype)
        z = z + (torch.matmul(x, w3.T) + b3) * mask3
        x = torch.relu(z)
        if return_stats:
            hidden_masks2_total = mask2.sum()
            hidden_masks3_total = mask3.sum()
            hidden_masks_den = mask2.numel()

        # Hidden layers
        chunk_size_w = self.hidden_dim * self.hidden_dim
        bias_size = self.hidden_dim
        for _ in range(len(self.hidden_layers)):
            w1 = interleaved_params[idx:idx+chunk_size_w].view(self.hidden_dim, self.hidden_dim)
            idx += chunk_size_w
            w2 = interleaved_params[idx:idx+chunk_size_w].view(self.hidden_dim, self.hidden_dim)
            idx += chunk_size_w
            w3 = interleaved_params[idx:idx+chunk_size_w].view(self.hidden_dim, self.hidden_dim)
            idx += chunk_size_w
            b1 = interleaved_params[idx:idx+bias_size]
            idx += bias_size
            b2 = interleaved_params[idx:idx+bias_size]
            idx += bias_size
            b3 = interleaved_params[idx:idx+bias_size]
            idx += bias_size

            z1 = torch.matmul(x, w1.T) + b1
            strength1 = self.activation_strength(z1)
            mask2 = (strength1 < act_threshold2).unsqueeze(1).to(z1.dtype)
            z = z1 + (torch.matmul(x, w2.T) + b2) * mask2
            strength2 = self.activation_strength(z)
            mask3 = (strength2 < act_threshold3).unsqueeze(1).to(z1.dtype)
            z = z + (torch.matmul(x, w3.T) + b3) * mask3
            x = torch.relu(z)
            if return_stats:
                hidden_masks2_total = hidden_masks2_total + mask2.sum()
                hidden_masks3_total = hidden_masks3_total + mask3.sum()
                hidden_masks_den = hidden_masks_den + mask2.numel()

        # Output layer
        chunk_size = self.output_layer.in_features * self.output_layer.out_features
        bias_size = self.output_layer.out_features
        w1 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape)
        idx += chunk_size
        w2 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape)
        idx += chunk_size
        w3 = interleaved_params[idx:idx+chunk_size].view(self.output_layer.weight.shape)
        idx += chunk_size
        b1 = interleaved_params[idx:idx+bias_size]
        idx += bias_size
        b2 = interleaved_params[idx:idx+bias_size]
        idx += bias_size
        b3 = interleaved_params[idx:idx+bias_size]
        # idx ends here

        y1 = torch.matmul(x, w1.T) + b1
        if final_threshold is None:
            final_threshold = self.default_uncertainty_threshold(final_metric)
        unc1 = self.compute_per_sample_uncertainty_from_logits(y1, metric=final_metric)
        mask2_out = (unc1 > final_threshold).unsqueeze(1).to(y1.dtype)
        y = y1 + (torch.matmul(x, w2.T) + b2) * mask2_out
        unc2 = self.compute_per_sample_uncertainty_from_logits(y, metric=final_metric)
        mask3_out = (unc2 > final_threshold).unsqueeze(1).to(y1.dtype)
        y = y + (torch.matmul(x, w3.T) + b3) * mask3_out
        if return_stats:
            out_expand2 = mask2_out.sum().item() / mask2_out.numel()
            out_expand3 = mask3_out.sum().item() / mask3_out.numel()
            hidden_expand2 = float(hidden_masks2_total.item()) / float(hidden_masks_den)
            hidden_expand3 = float(hidden_masks3_total.item()) / float(hidden_masks_den)
            return y, {"out_expand2_rate": out_expand2, "out_expand3_rate": out_expand3, "hidden_expand2_rate": hidden_expand2, "hidden_expand3_rate": hidden_expand3}
        return y

# Compute parameters
model = CompressedMLP()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total original parameters: {total_params}")
total_expanded = total_params * 3
print(f"Total expanded parameters: {total_expanded}")

# Verify lossless recovery (pre-training)
with torch.no_grad():
    orig_weights = [p.clone() for p in model.parameters()]
    interleaved = model.interleave_params()
    model.deinterleave_params(interleaved)
    max_error = max(torch.abs(p - orig).max().item() for p, orig in zip(model.parameters(), orig_weights))
print(f"Max error after round-trip (pre-training): {max_error:.2e}")

# Data loading (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train with expanded forward and auxiliary band-1; band-3 regularizer (1 epoch)
model.train()
interleaved_params = nn.Parameter(model.interleave_params().to(device))
optimizer_inter = optim.SGD([interleaved_params], lr=0.01)
aux_weight = 0.1
reg_weight = 1e-6
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer_inter.zero_grad()
    logits_full = model.forward_expanded(data, interleaved_params)
    loss_full = criterion(logits_full, target)
    logits_band1 = model.forward_banded(data, interleaved_params, bands=1)
    loss_aux = criterion(logits_band1, target)
    reg = model.compute_band3_l2(interleaved_params)
    loss = loss_full + aux_weight * loss_aux + reg_weight * reg
    loss.backward()
    optimizer_inter.step()
    model.deinterleave_params(interleaved_params)
    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Aux: {loss_aux.item():.4f}")

# Verify lossless recovery (post-training, no updates for test)
with torch.no_grad():
    interleaved = model.interleave_params()
    model.deinterleave_params(interleaved)
    max_error = max(torch.abs(p - orig).max().item() for p, orig in zip(model.parameters(), orig_weights))
print(f"Max error after round-trip (post-training, no updates): {max_error:.2e}")

# Calibrate output threshold to meet target expansion rate
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    interleaved_eval = model.interleave_params().to(device)
    # Simple calibration: scan thresholds and choose one that achieves target expansion rate on output band-2
    target_expand_rate = 0.1
    uncertainty_metric = "entropy"  # or "one_minus_max_prob"
    candidate_thresholds = torch.linspace(0.2, 1.8, steps=9)
    best_thr = None
    best_diff = 1e9
    # Use a small calibration subset
    calib_seen = 0
    max_calib = 3000
    expand_rates = []
    for thr in candidate_thresholds:
        expanded = 0
        total = 0
        calib_seen = 0
        for data, target in test_loader:
            data = data.to(device)
            y1 = model.forward_banded(data, interleaved_eval, bands=1)
            unc = model.compute_per_sample_uncertainty_from_logits(y1, metric=uncertainty_metric)
            expanded += (unc > float(thr.item())).sum().item()
            total += unc.numel()
            calib_seen += data.size(0)
            if calib_seen >= max_calib:
                break
        rate = expanded / max(1, total)
        expand_rates.append((float(thr.item()), rate))
        diff = abs(rate - target_expand_rate)
        if diff < best_diff:
            best_diff = diff
            best_thr = float(thr.item())
    threshold = best_thr if best_thr is not None else model.default_uncertainty_threshold(uncertainty_metric)
    print(f"Calibrated threshold ({uncertainty_metric}): {threshold:.3f}; rates={expand_rates}")

# Evaluate with progressive gating and stats
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    interleaved_eval = model.interleave_params().to(device)
    expand2_total = 0.0
    expand3_total = 0.0
    hidden2_total = 0.0
    hidden3_total = 0.0
    batches = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output, stats = model.forward_progressive(data, interleaved_eval, final_metric=uncertainty_metric, final_threshold=threshold, return_stats=True)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        expand2_total += stats["out_expand2_rate"]
        expand3_total += stats["out_expand3_rate"]
        hidden2_total += stats["hidden_expand2_rate"]
        hidden3_total += stats["hidden_expand3_rate"]
        batches += 1

test_loss /= len(test_loader)
accuracy = 100. * correct / len(test_dataset)
print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
if batches > 0:
    print(f"Avg output expand2 rate: {expand2_total / batches:.3f}, expand3 rate: {expand3_total / batches:.3f}")
    print(f"Avg hidden expand2 rate: {hidden2_total / batches:.3f}, hidden expand3 rate: {hidden3_total / batches:.3f}")