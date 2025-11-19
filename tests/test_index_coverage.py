import torch
from imc import CompressedMLP
from imc.infer_cpu import forward_progressive_bucketing


def test_index_coverage_for_forwards():
    torch.manual_seed(0)
    model = CompressedMLP(hidden_dim=16, num_hidden_layers=3)
    inter = model.interleave_params()
    x = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        y_b1 = model.forward_banded(x, inter, bands=1)
        y_b3 = model.forward_banded(x, inter, bands=3)
        y_exp = model.forward_expanded(x, inter)
        y_prog = forward_progressive_bucketing(model, x, inter, return_stats=False)
    assert y_b1.shape == y_b3.shape == y_exp.shape == y_prog.shape


