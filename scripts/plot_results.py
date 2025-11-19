import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="artifacts/bench_cpu_results.json")
    parser.add_argument("--out", type=str, default="artifacts/plots")
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    timings = data["timing_s_per_iter"]
    acc = data["accuracy_pct"]

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Pareto (latency vs accuracy)
    modes = ["merged", "expanded", "progressive", "early_exit"]
    xs = [timings[m] for m in modes]
    ys = [acc[m] for m in modes]
    plt.figure()
    plt.scatter(xs, ys)
    for i, m in enumerate(modes):
        plt.annotate(m, (xs[i], ys[i]))
    plt.xlabel("Latency (s/iter)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Latency (CPU)")
    plt.grid(True, alpha=0.3)
    plt.savefig(str(Path(args.out) / "pareto_cpu.png"), dpi=200, bbox_inches="tight")
    print(f"Saved {Path(args.out) / 'pareto_cpu.png'}")


if __name__ == "__main__":
    main()


