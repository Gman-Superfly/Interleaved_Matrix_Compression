import torch
from imc import MatrixCompressor, CompressedMLP


def test_matrix_roundtrip_exact_on_lattice():
    torch.manual_seed(0)
    comp = MatrixCompressor()
    W = torch.randn(8, 8)
    W_snap = comp.warmup(W.clone())
    m1, m2, m3 = comp.split(W_snap)
    W_rec = comp.merge(m1, m2, m3)
    assert torch.allclose(W_rec, W_snap, atol=0.0, rtol=0.0)


def test_model_interleave_deinterleave_roundtrip():
    torch.manual_seed(0)
    model = CompressedMLP(hidden_dim=32, num_hidden_layers=4)
    with torch.no_grad():
        orig = [p.clone() for p in model.parameters()]
        inter = model.interleave_params()
        model.deinterleave_params(inter)
        errs = [torch.max(torch.abs(p - o)).item() for p, o in zip(model.parameters(), orig)]
    assert max(errs) == 0.0


