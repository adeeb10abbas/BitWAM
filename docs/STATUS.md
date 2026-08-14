# Execution status

Status recorded on 2026-08-14 from branch `ali/claude`.

## Verified locally

- Phase 1 acceptance: clean Python 3.12 package, exact `uv.lock`, import,
  `bitwam --help`, and CPU tests.
- Phase 2 implementation: registered wrapper, upstream processor reuse, native
  source loading, exact seeded CPU action parity, and CPU forward/backward.
- Phase 3 acceptance: BF16 master weights, ternary/INT8 simulation, STE update,
  deterministic conversion boundaries, recovery boundaries, save/load, and
  quantization coverage reporting.
- Phase 4 runner: baseline, primary pilot, and the two predefined recovery paths
  translate to the pinned LeRobot commands and persist PID/log/state metadata.

The current check result is `19 passed, 1 skipped`. The skipped test is the CUDA
forward/backward smoke test.

## Hardware blocker

This host exposes no `/dev/nvidia*`, and `nvidia-smi` cannot communicate with an
NVIDIA driver. Therefore none of the following has run:

- the CUDA BF16 acceptance smoke;
- the real `lerobot/VLA-JEPA-LIBERO` baseline;
- either 2,000-step QAT pilot or its 50 closed-loop LIBERO-10 episodes;
- packed Triton validation and GPU benchmarks;
- the 12-run final training matrix or its evaluations.

No task-success, compression, VRAM, or latency claim is currently authorized.

## Resume on an Ampere-or-newer GPU

```bash
git switch ali/claude
uv sync --frozen
uv run pytest -q
uv run pytest -q tests/test_policy.py::test_cuda_forward_backward_smoke
uv run bitwam evaluate --config configs/baseline.yaml
uv run bitwam train --config configs/pilot.yaml
uv run bitwam evaluate --config configs/pilot.yaml
```

Continue to `configs/pilot_qwen_dit.yaml` only when the Qwen pilot reaches at
least 90% of the measured BF16 success rate and remains numerically stable. Use
each recovery config at most once, exactly as specified in the execution plan.
