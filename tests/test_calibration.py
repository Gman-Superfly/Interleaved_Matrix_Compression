import torch
from torch.utils.data import DataLoader, TensorDataset
from imc import CompressedMLP
from imc.calibrate import compute_hidden_strengths_percentiles, calibrate_output_threshold


def make_fake_loader(n: int = 256, batch: int = 64):
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch, shuffle=False)


def test_compute_hidden_strengths_percentiles_outputs_shapes():
    torch.manual_seed(0)
    model = CompressedMLP(hidden_dim=16, num_hidden_layers=3)
    inter = model.interleave_params()
    loader = make_fake_loader()
    out = compute_hidden_strengths_percentiles(model, loader, inter, percentiles=(0.6, 0.8), max_batches=2)
    assert "hidden_thr2" in out and "hidden_thr3" in out
    layers_total = 1 + len(model.hidden_layers)
    assert out["hidden_thr2"].shape[0] == layers_total
    assert out["hidden_thr3"].shape[0] == layers_total


def test_calibrate_output_threshold_returns_float():
    torch.manual_seed(0)
    model = CompressedMLP(hidden_dim=16, num_hidden_layers=3)
    inter = model.interleave_params()
    loader = make_fake_loader()
    thr = calibrate_output_threshold(model, loader, inter, target_expand_rate=0.2, metric="entropy", max_samples=128)
    assert isinstance(thr, float)
    assert thr > 0.0


