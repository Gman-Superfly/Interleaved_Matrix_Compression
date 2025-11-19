---
title: "Interleaved Matrix Compression: Lossless bands for dynamic inference"
author: "Datamutant.ai"
date: "2025-11-19"
---

Why compute more than you need? IMC keeps weights in a compact, reversible form and unfolds
additional detail on demand, guided by uncertainty. This post explains the idea, shows a CPU demo
on MNIST, and shares an accuracy–latency Pareto.

## TL;DR
- Lossless banded split with fp32-lattice guarantees
- Single-pass progressive inference with per-sample gating
- Real CPU savings via sample bucketing (no full-batch masked matmuls)

## Quickstart
```
pip install -r requirements.txt
python examples/roundtrip_minimal.py
python examples/progressive_demo_cpu.py
```

## How it works
Bit-plane inspired split; integer-bounded bands; output entropy decides whether to unfold further.

## Results (CPU, MNIST)
See `artifacts/plots/pareto_cpu.png`.


