## Interleaved Matrix Compression (IMC)

IMC explores a simple but powerful idea: compression is not an afterthought, it is the baseline. We keep the model in a compact, lossless state by default and unfold extra detail only when uncertainty justifies the extra compute.

This repository implements a reversible, banded decomposition of neural network weights, plus a single-pass, on-the-fly unfolding mechanism driven by uncertainty signals. The result is a model that is efficient on easy inputs and selectively spends compute only where it matters.

### Why this matters
- **Compression-as-baseline**: Efficient systems settle to an incompressible core. Our weights live in a compact, exact representation; unfolding is temporary and targeted.
- **Dynamic efficiency**: Most inputs are “easy.” We compute with a cheap approximation (band-1) and add refinement bands only for uncertain inputs.
- **Safety and guarantees**: Splitting and merging are lossless at fp32 mantissa precision. The fully expanded path reproduces the exact merged model.

## Core concepts

### Lossless, banded weight decomposition
- Each parameter tensor is split into three integer-bounded “bands” that sum exactly to the original tensor when rescaled.
- Scaling uses `2**24` so integer operations are exact within fp32 mantissa. Negatives are handled symmetrically.
- The split is controlled by two cumulative boundaries (keys), which define per-band capacities for band-1 and band-2; band-3 stores the remainder.

Properties:
- **Exactness**: `merge(split(W)) == W` within fp32 tolerance.
- **Coarse-to-fine**: band-1 tends to carry coarse structure, band-2 refines it, band-3 captures residuals.

### Three forward paths (all available)
- **Merged**: Standard layers with normal weights. Fastest to write, used as a baseline check.
- **Expanded**: Explicitly sums all three bands per layer. Algebraically equals the merged path (full fidelity). Used for training of the interleaved parameter vector.
- **Progressive (preferred at inference)**: Single pass with per-layer gating. Starts with band-1; conditionally adds band-2 and band-3 contributions based on uncertainty. No re-running.

### Uncertainty-driven unfolding
- Hidden layers use a cheap, per-sample activation-strength signal to decide whether to add the next band.
- The output layer uses a classification uncertainty metric (entropy or `1 - max_prob`) with a calibrated threshold.
- Thresholds can be tuned to achieve a target expansion rate (e.g., ~10%).

### Training that favors the compressed path
- The model trains the fully expanded path for accuracy.
- An auxiliary loss on the band-1-only path strengthens the compressed baseline.
- A small L2 regularizer on band-3 pushes information toward earlier bands (“nested sparsity”).

## What’s in the code

- `imc.py`
  - `MatrixCompressor`: lossless split/merge with symmetric handling of negatives and vectorized ops; warmup to snap weights onto the fp32 grid.
  - `CompressedMLP`: large MLP with:
    - Parameter warmup on init.
    - Interleave/deinterleave to flatten and restore all bands.
    - `forward_expanded` (full fidelity), `forward` (merged), and `forward_progressive` (single-pass unfolding with gates).
    - Uncertainty utilities: batch and per-sample metrics + default thresholds.
    - `forward_banded` for band-limited passes (bands=1/2/3) used in training and calibration.
    - `compute_band3_l2` for the regularizer.
    - Validation helpers: `_validate_interleaved_vector` + `expected_interleaved_numel()` ensure 1D interleaved vector length matches the architecture.
    - Lightweight assertions check index coverage after forward/interleave operations to catch slicing mismatches early.
    - Type hints across public methods for clarity and tooling support.
  - Implementation notes:
    - Device-safe compressor: scale factors move to the same device/dtype as tensors to avoid cross-device ops.
  - Training loop: full loss + auxiliary band-1 loss + small band-3 L2.
  - Calibration pass: scans output thresholds to hit a target expansion rate.
  - Evaluation: uses `forward_progressive` with calibrated threshold and reports expansion statistics.

- `imc/` (package)
  - `compress.py`: `MatrixCompressor` (library form)
  - `model.py`: `CompressedMLP` (library form)
  - `infer_cpu.py`: `forward_progressive_bucketing` — compute-efficient progressive path on CPU (per-sample bucketing; no full-batch masked matmuls)
  - `calibrate.py`: hidden-layer percentile calibration + output threshold calibration; save/load JSON
  - `train_cpu.py`: tiny training utilities (MNIST)

- Scripts and examples
  - `scripts/bench_cpu.py`: accuracy/latency benchmarking (merged vs expanded vs progressive vs static early-exit)
  - `scripts/plot_results.py`: Pareto plot
  - `examples/roundtrip_minimal.py`: minimal split/merge demo
  - `examples/progressive_demo_cpu.py`: progressive inference with stats and JSON thresholds

## Guarantees and invariants
- Round-trip invariance: `deinterleave(interleave(params))` preserves weights (pre/post training checks included).
- Lossless decomposition: `merge(m1, m2, m3) == original` within fp32 tolerance.
- Full equivalence: `forward_expanded` equals merged when all three bands are included.

## Validation and typing
- Assert-early checks verify:
  - Interleaved vector is 1D and has the exact expected length.
  - Forward passes and readers consume the entire interleaved vector (index coverage).
- Methods include type hints; simple guards ensure inputs are Tensors with expected shapes.

## Running locally

### Requirements
- Python 3.10+
- PyTorch and TorchVision (CPU or CUDA builds)

Install dependencies (Windows PowerShell):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Quick demos (CPU):

```powershell
# Minimal round-trip
python examples/roundtrip_minimal.py

# Progressive inference (calibration + stats)
python examples/progressive_demo_cpu.py

# Benchmark + Pareto plot
python scripts/bench_cpu.py --hidden-dim 128 --layers 16 --device cpu
python scripts/plot_results.py --results artifacts/bench_cpu_results.json
```

Notes:
- The provided MLP is intentionally deep (default 380 layers) and may be slow; feel free to reduce `num_hidden_layers` and/or `hidden_dim` in `CompressedMLP` for quick iteration.
- If you have CUDA, PyTorch will use it automatically.
 - The example calibration scans thresholds using the test loader for brevity; for proper evaluation, prefer a held-out validation split for calibration.
 - Calibrated thresholds (CPU/MNIST) are saved to `artifacts/thresholds_cpu_mnist.json`.

## Key knobs to tune
- In `forward_progressive` (inference):
  - `act_threshold2`, `act_threshold3`: hidden-layer activation gates for adding band-2/band-3.
  - `final_metric`, `final_threshold`: output uncertainty and calibrated threshold.
- In training:
  - `aux_weight`: strength of band-1 auxiliary loss.
  - `reg_weight`: L2 weight on band-3.
  - Optimizer/learning rate if you swap from SGD to AdamW, etc.
- In calibration:
  - `target_expand_rate` and candidate threshold grid.

## Design choices and rationale
- **Single-pass unfolding**: Gating happens inside each layer; no second pass through the network. This matches the “unfold on demand” principle.
- **Lossless by construction**: Integer-bounded bands with `2**24` scaling keep a reliable, reversible core.
- **Auxiliary/regularized training**: Encourages information to concentrate in early bands, making the compressed path both fast and accurate.
- **Simple gates first**: Activation-strength for hidden layers keeps overhead small; the final layer uses a better-calibrated classification metric.

## Gaps/risks to watch

- **Scale invariance of hidden-layer gates**: Activation-strength can be input-scale sensitive; consider normalizing (e.g., post-layernorm statistics) to make thresholds portable.
- **Numerical edge cases**: Explicitly document handling for near-limit values after `2**24` scaling and negatives; include saturation/overflow checks in tests.
- **Train/infer mismatch**: If training never experiences gating decisions, distribution shift can creep in; lightweight gating-aware training or stochastic gating probing may help.
- **Defaults for quick start**: A 380-layer MLP is heavy; expose a "tiny" config in the README for CPU users to validate in seconds.

## Quick wins to strengthen README

- **Minimal, copy-paste example**: Show `MatrixCompressor.split/merge` round-trip on a small tensor and a one-batch `forward_progressive` run that prints expansion rate.
- **Calibration snippet**: One short code sample that scans thresholds to hit a target expansion rate and saves them, then uses them at eval.
- **Guarantees section with bounds**: One line on why `2**24` suffices for fp32 mantissa and how remainder ensures exactness within fp32 tolerance.

## Extensions we consider next
- Generalize to N bands (keys → K cumulative boundaries) with fully vectorized split/merge.
- Replace activation-strength gates with tiny per-layer probe heads for more principled mid-network uncertainty.
- Learn per-layer band boundaries (with constraints) while keeping losslessness via the remainder band.
- Real storage compression: cast bands to smaller integer widths and/or entropy-code for disk/transfer.
- Apply to other architectures (CNNs, Transformers) to validate generality.

## TODO

- **Unit tests**: Round-trip invariance, interleaved-vector length, 100% index coverage, expanded==merged parity, and gating expansion-rate targets.
- **Instrumentation**: Log per-layer expansion histograms and end-to-end compute savings; add a "target 10% expansion" preset with saved thresholds.

## License

MIT License. See `LICENSE`.


