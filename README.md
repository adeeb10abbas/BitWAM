# BitWAM

BitWAM is a LeRobot policy plugin for measuring how ternary compression affects
VLA-JEPA action generation and world-model supervision.

The native BitVLA path also includes a quality-gated packed runtime. The recommended
mode packs the ternary text backbone into four two-bit codes per byte, preserves
bit-exact actions and 10/10 ordered LIBERO smoke success, and reduces measured B200
resident CUDA allocation by 62.08%. See
[`PACKED_RUNTIME.md`](results/native-bitvla-seed0-50/PACKED_RUNTIME.md) for the full
memory, latency, and ablation results.

The implementation follows [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md).
The research claims and matched ablations are defined in
[`docs/PAPER_PLAN.md`](docs/PAPER_PLAN.md).
The living manuscript is in [`docs/PAPER_DRAFT.md`](docs/PAPER_DRAFT.md); pending
result cells are filled only from archived closed-loop artifacts.
Large datasets, checkpoints, simulator output, and videos stay outside Git.

## Setup

BitWAM targets Linux, Python 3.12, and Ampere-or-newer NVIDIA GPUs. The lockfile
pins an exact LeRobot 0.6.2 development revision and its CUDA 12.8 PyTorch stack.

```bash
uv sync
uv run bitwam --help
uv run pytest -q
```

## Commands

Every workflow uses one YAML configuration file:

```bash
uv run bitwam evaluate --config configs/baseline.yaml
uv run bitwam train --config configs/pilot.yaml
uv run bitwam evaluate --config configs/pilot.yaml
uv run bitwam export --config configs/pilot.yaml       # Phase 5
uv run bitwam benchmark --config configs/pilot.yaml    # Phase 5
uv run bitwam summarize --config configs/final_matrix.yaml
```

For the selected native BitVLA packed runtime:

```bash
uv run bitwam benchmark --config configs/bitvla-world-ternary-packed-text-benchmark.yaml
uv run bitwam evaluate --config configs/bitvla-world-ternary-packed-text-eval-10.yaml
```

Training writes its PID, exact command, status JSON, and append-only log beside
the configured output path so LeRobot can safely create that path itself.
Evaluation stores the same metadata under the run path. Re-running an identical
completed command skips it. Use a recovery config only if the corresponding
first pilot misses its gate; the recovery choices are deliberately fixed by the
execution plan.

The GPU-backed pilot and final matrix must not be reported as complete without
their closed-loop LIBERO artifacts and manifests.

See [`docs/STATUS.md`](docs/STATUS.md) for the latest verified gate state.
