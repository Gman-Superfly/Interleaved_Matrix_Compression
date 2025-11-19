import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from imc import CompressedMLP
from imc.infer_gpu import forward_progressive_bucketing
from imc.calibrate import compute_hidden_strengths_percentiles, calibrate_output_threshold, save_thresholds_json


def build_cifar_loaders(batch_size: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="artifacts")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    _, test_loader = build_cifar_loaders()

    model = CompressedMLP(input_dim=3 * 32 * 32, hidden_dim=args.hidden_dim, output_dim=10, num_hidden_layers=args.layers).to(device)
    interleaved = model.interleave_params().to(device)
    criterion = nn.CrossEntropyLoss()

    # Simple calibration
    hidden_thrs = compute_hidden_strengths_percentiles(model, test_loader, interleaved, percentiles=(0.7, 0.85), max_batches=30)
    out_thr = calibrate_output_threshold(model, test_loader, interleaved, target_expand_rate=0.1, metric="entropy")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    save_thresholds_json(str(Path(args.outdir) / "thresholds_gpu_cifar10.json"), out_thr, hidden_thrs)

    # Evaluate progressive
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = forward_progressive_bucketing(
                model, data, interleaved,
                final_metric="entropy",
                final_threshold=out_thr,
                return_stats=False,
                per_layer_thresholds=hidden_thrs,
            )
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
    acc = 100.0 * correct / max(1, total)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.outdir, "cifar_eval.json").write_text(json.dumps({"accuracy": acc}, indent=2))
    print(json.dumps({"accuracy": acc}, indent=2))


if __name__ == "__main__":
    main()


