import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from imc import CompressedMLP
from imc.infer_gpu import forward_progressive_bucketing
from imc.calibrate import compute_hidden_strengths_percentiles, calibrate_output_threshold, save_thresholds_json


def build_cifar_loader(batch_size_test: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    return test_dataset, test_loader


def time_eval(fn, iters: int = 20):
    # Warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    return (t1 - t0) / max(1, iters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="artifacts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = build_cifar_loader()

    model = CompressedMLP(input_dim=3*32*32, hidden_dim=args.hidden_dim, output_dim=10, num_hidden_layers=args.layers).to(device)
    interleaved = model.interleave_params().to(device)
    criterion = nn.CrossEntropyLoss()

    # Calibrate thresholds
    hidden_thrs = compute_hidden_strengths_percentiles(model, test_loader, interleaved, percentiles=(0.7, 0.85), max_batches=30)
    out_thr = calibrate_output_threshold(model, test_loader, interleaved, target_expand_rate=0.1, metric="entropy")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    save_thresholds_json(str(Path(args.outdir) / "thresholds_gpu_cifar10.json"), out_thr, hidden_thrs)

    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)

    def run_merged():
        with torch.no_grad():
            return model(data)
    def run_expanded():
        with torch.no_grad():
            return model.forward_expanded(data, interleaved)
    def run_progressive():
        with torch.no_grad():
            return forward_progressive_bucketing(model, data, interleaved, final_metric="entropy", final_threshold=out_thr, return_stats=False, per_layer_thresholds=hidden_thrs)

    t_merged = time_eval(run_merged, iters=args.iters)
    t_expanded = time_eval(run_expanded, iters=args.iters)
    t_progressive = time_eval(run_progressive, iters=args.iters)

    results = {
        "timing_s_per_iter": {"merged": t_merged, "expanded": t_expanded, "progressive": t_progressive},
        "config": {"hidden_dim": args.hidden_dim, "layers": args.layers, "device": str(device), "iters": args.iters},
    }
    out_path = Path(args.outdir) / "bench_gpu_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


