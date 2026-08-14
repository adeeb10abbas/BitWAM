# BitWAM

BitWAM is a LeRobot policy plugin for measuring how ternary compression affects
VLA-JEPA action generation and world-model supervision.

The implementation follows [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md).
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

Training and evaluation write a PID, exact command, status JSON, and append-only
log under the configured `output_dir`. Re-running an identical completed command
skips it. Use a recovery config only if the corresponding first pilot misses its
gate; the recovery choices are deliberately fixed by the execution plan.

The GPU-backed pilot and final matrix must not be reported as complete without
their closed-loop LIBERO artifacts and manifests.

See [`docs/STATUS.md`](docs/STATUS.md) for the latest verified gate state.
