import torch
from imc import CompressedMLP


def test_forward_expanded_equals_merged_after_deinterleave():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = CompressedMLP(hidden_dim=32, num_hidden_layers=4).to(device)
    inter = model.interleave_params().to(device)
    # Set module weights to merge of split weights
    model.deinterleave_params(inter)
    x = torch.randn(16, 1, 28, 28, device=device)
    with torch.no_grad():
        y_merged = model(x)
        y_expanded = model.forward_expanded(x, inter)
    assert torch.allclose(y_merged, y_expanded, atol=0.0, rtol=0.0)


