import torch
from imc import MatrixCompressor


def main():
    torch.manual_seed(0)
    W = torch.randn(4, 4)
    comp = MatrixCompressor()
    W_snap = comp.warmup(W.clone())
    m1, m2, m3 = comp.split(W_snap)
    W_rec = comp.merge(m1, m2, m3)
    print("max_abs_error:", float((W_rec - W_snap).abs().max()))


if __name__ == "__main__":
    main()


