import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from imc import CompressedMLP, forward_progressive_bucketing
from imc.calibrate import compute_hidden_strengths_percentiles, calibrate_output_threshold, save_thresholds_json


def build_mnist_loaders(batch_size_test: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    return test_dataset, test_loader


def time_eval(fn, iters: int = 20):
    # Warmup
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / max(1, iters)


def forward_partial(model: CompressedMLP, x: torch.Tensor, layers: int) -> torch.Tensor:
    """Static early-exit baseline: run only first `layers` hidden layers (no banding)."""
    x = x.view(x.size(0), -1)
    x = torch.relu(model.input_layer(x))
    for i, layer in enumerate(model.hidden_layers):
        if i >= layers:
            break
        x = torch.relu(layer(x))
    return model.output_layer(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="artifacts")
    parser.add_argument("--calib-batches", type=int, default=30)
    parser.add_argument("--percentiles", type=str, default="0.7,0.85")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))
    test_dataset, test_loader = build_mnist_loaders()

    model = CompressedMLP(hidden_dim=args.hidden_dim, num_hidden_layers=args.layers).to(device)
    interleaved = model.interleave_params().to(device)
    criterion = nn.CrossEntropyLoss()

    # Calibrate thresholds (hidden via percentiles, output via target expand rate)
    p2, p3 = [float(x) for x in args.percentiles.split(",")]
    hidden_thrs = compute_hidden_strengths_percentiles(model, test_loader, interleaved, percentiles=(p2, p3), max_batches=args.calib_batches)
    out_thr = calibrate_output_threshold(model, test_loader, interleaved, target_expand_rate=0.1, metric="entropy")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    save_thresholds_json(str(Path(args.outdir) / "thresholds_cpu_mnist.json"), out_thr, hidden_thrs)

    # Collect a single batch for timing
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)

    # Baselines
    def run_merged():
        with torch.no_grad():
            return model(data)

    def run_expanded():
        with torch.no_grad():
            return model.forward_expanded(data, interleaved)

    def run_progressive():
        with torch.no_grad():
            return forward_progressive_bucketing(
                model, data, interleaved,
                final_metric="entropy",
                final_threshold=out_thr,
                return_stats=False,
                per_layer_thresholds=hidden_thrs,
            )

    def run_early_exit():
        with torch.no_grad():
            # Exit at half the layers
            return forward_partial(model, data, layers=max(1, args.layers // 2))

    t_merged = time_eval(run_merged, iters=args.iters)
    t_expanded = time_eval(run_expanded, iters=args.iters)
    t_progressive = time_eval(run_progressive, iters=args.iters)
    t_early = time_eval(run_early_exit, iters=args.iters)

    # Accuracy estimates (single pass over test set per mode)
    def eval_mode(forward_fn):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = forward_fn(data)
                preds = logits.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += data.size(0)
        return 100.0 * correct / max(1, total)

    acc_merged = eval_mode(lambda d: model(d))
    acc_expanded = eval_mode(lambda d: model.forward_expanded(d, interleaved))
    acc_progressive = eval_mode(lambda d: forward_progressive_bucketing(model, d, interleaved, final_metric="entropy", final_threshold=out_thr, return_stats=False, per_layer_thresholds=hidden_thrs))
    acc_early = eval_mode(lambda d: forward_partial(model, d, layers=max(1, args.layers // 2)))

    results = {
        "timing_s_per_iter": {
            "merged": t_merged,
            "expanded": t_expanded,
            "progressive": t_progressive,
            "early_exit": t_early,
        },
        "accuracy_pct": {
            "merged": acc_merged,
            "expanded": acc_expanded,
            "progressive": acc_progressive,
            "early_exit": acc_early,
        },
        "config": {
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "device": str(device),
            "iters": args.iters,
            "percentiles": [p2, p3],
        },
    }
    out_path = Path(args.outdir) / "bench_cpu_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


