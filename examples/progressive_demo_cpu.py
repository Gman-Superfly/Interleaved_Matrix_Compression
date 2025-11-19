import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from imc import CompressedMLP, forward_progressive_bucketing
from imc.calibrate import compute_hidden_strengths_percentiles, calibrate_output_threshold, save_thresholds_json


def main():
    device = torch.device("cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = CompressedMLP(hidden_dim=128, num_hidden_layers=16).to(device)
    interleaved = model.interleave_params().to(device)

    hidden_thrs = compute_hidden_strengths_percentiles(model, test_loader, interleaved, percentiles=(0.7, 0.85), max_batches=20)
    out_thr = calibrate_output_threshold(model, test_loader, interleaved, target_expand_rate=0.1, metric="entropy")
    save_thresholds_json("artifacts/thresholds_cpu_mnist.json", out_thr, hidden_thrs)

    # One batch inference with stats
    data, target = next(iter(test_loader))
    data = data.to(device)
    with torch.no_grad():
        logits, stats = forward_progressive_bucketing(
            model, data, interleaved,
            final_metric="entropy",
            final_threshold=out_thr,
            return_stats=True,
            per_layer_thresholds=hidden_thrs,
        )
    print("Stats:", stats)
    preds = logits.argmax(dim=1)
    print("Preds shape:", preds.shape)


if __name__ == "__main__":
    main()


