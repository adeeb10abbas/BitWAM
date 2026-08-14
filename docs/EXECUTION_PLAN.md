# BitWAM Execution Plan

## Goal

Replace the unfinished prototype with a small LeRobot policy plugin built on pretrained VLA-JEPA. First establish a reliable BF16 baseline, then progressively ternarize the deployed policy, add genuinely packed inference, and run the minimum research-grade experiment needed to show whether world-model supervision survives compression.

The project is successful when BitWAM retains at least 95% of BF16 task success, shows a positive world-loss benefit after ternarization, and stores at least 80% of eligible inference weights in packed ternary form.

## Execution rules

- Work through the phases in order and make one small commit after each completed phase.
- Favor a working, measurable result over extra abstractions or broad audits.
- Do not redesign the architecture unless a stated acceptance gate cannot be met.
- Allow at most one predefined recovery attempt at each quantization stage.
- Keep datasets, checkpoints, rollout videos, and other large artifacts outside Git.
- Record exact dependency versions, upstream commits, configs, seeds, and checkpoint identifiers.

## Phase 1: Clean and rename the codebase

- Preserve Git history and the license; delete the old implementation, archive, scripts, tests, and speculative reports.
- Create a Python 3.12 `uv` project named `lerobot-policy-bitwam`, with source package `lerobot_policy_bitwam`.
- Pin LeRobot 0.6.2 and its compatible PyTorch/CUDA stack in `uv.lock`.
- Keep the repository small: concise README, AGENTS.md, experiment configs, package code, focused tests, and this plan.
- Add a single `bitwam` command with `train`, `evaluate`, `export`, `benchmark`, and `summarize` subcommands.

Acceptance gate: the package imports in a clean environment, `bitwam --help` works, and CPU unit tests can start without downloading model weights.

## Phase 2: Establish the BF16 baseline

- Implement `BitWAMPolicy` as a thin registered LeRobot policy around upstream `VLAJEPAPolicy`.
- Reuse VLA-JEPA preprocessing, checkpoint loading, world loss, action generation, and evaluation behavior.
- Support `lerobot/VLA-JEPA-LIBERO` for immediate evaluation and `lerobot/VLA-JEPA-Pretrain` for controlled experiments.
- Implement the LeRobot methods `reset`, `select_action`, `predict_action_chunk`, and training `forward`.
- Add a quantization-disabled configuration that behaves like upstream VLA-JEPA.

Acceptance gate: with identical inputs and weights, quantization-disabled BitWAM matches upstream VLA-JEPA actions within BF16 tolerance and completes one CUDA forward/backward smoke test.

## Phase 3: Add training-time ternarization

- Implement `TernaryLinear` with BF16 master weights, per-output-channel abs-mean scaling, ternary values `{-1, 0, +1}`, straight-through gradients, and per-token INT8 activation simulation.
- Implement deterministic conversion scopes:
  - `none`: normal BF16.
  - `qwen`: Qwen attention and MLP linear layers.
  - `qwen_dit`: Qwen plus DiT transformer attention and MLP layers.
- Keep embeddings, normalization, vision projections, state/action/time encoders, and final action output layers in BF16.
- Implement `convert_for_qat(policy, scope) -> QuantizationReport` and report total, eligible, ternary, and BF16 parameter counts.
- Test ternary values and scales, nonzero STE gradients, conversion boundaries, save/load, and one training update.

Acceptance gate: converted weights receive nonzero gradients, a training step changes BF16 master weights, and the conversion report agrees with the actual module tree.

## Phase 4: Run the short pilot

- Measure the unmodified `lerobot/VLA-JEPA-LIBERO` checkpoint first.
- Run one seed of Qwen-only QAT for 2,000 steps, then evaluate 50 episodes distributed across LIBERO-10.
- Continue if task success is at least 90% of BF16 and training remains numerically stable.
- If needed, make one recovery attempt using a lower learning rate while leaving the first and last Qwen blocks in BF16.
- Repeat the gated pilot for `qwen_dit`.
- Its single allowed recovery keeps the final four DiT blocks in BF16.
- If either recovery still misses the gate, stop expanding quantization and document the result instead of running an open-ended search.

## Phase 5: Add packed inference

- Export four ternary values per byte, BF16 per-output-channel scales, and separate BF16 fallback tensors.
- Provide a portable `reference` backend for correctness and a `triton` backend for Ampere-or-newer NVIDIA GPUs.
- The Triton path must consume packed weights directly. Unpacking the whole model to BF16 at startup does not qualify as cheaper inference.
- Compare packed output with the reference backend using a fixed numerical tolerance.
- Benchmark end-to-end batch-1 inference after warmup and record serialized size, peak VRAM, and p50/p95 latency.

Acceptance gate: packed inference passes numerical checks and materially reduces serialized deployed-model size and peak inference VRAM. Report latency honestly whether it improves or not.

## Phase 6: Run the final experiment

Train from `lerobot/VLA-JEPA-Pretrain` for 30,000 steps using seeds `0`, `1`, and `2`:

1. BF16 with world loss `0.1`.
2. BF16 without world loss.
3. Ternary with world loss `0.1`.
4. Ternary without world loss.

Use BF16 compute and an effective global batch size of 64. Set gradient accumulation from GPU count while preserving that effective batch size.

Evaluate each run on all ten LIBERO-10 tasks with 50 episodes per task. Record:

- Mean and per-task closed-loop success.
- World representation prediction loss.
- Percentage of eligible deployed weights that are ternary.
- Serialized deployed-model size.
- Peak inference VRAM.
- Batch-1 p50 and p95 latency.

Final acceptance criteria:

- Ternary-with-world success is at least 95% of BF16-with-world success.
- World supervision improves mean ternary task success across the three seeds.
- At least 80% of eligible deployed-policy weights are ternary.
- Packed model size and peak VRAM improve materially.
- All latency measurements are reported, including regressions.

## Phase 7: Package the result

- Store run metadata in `results/<run-id>/manifest.json` and summary metrics in `metrics.json`.
- Generate one compact results table with per-seed values and aggregate mean/std.
- Document one setup command, one smoke-test command, one pilot command, and one final-matrix command.
- Commit configs, the environment lock, metric summaries, and exact checkpoint/LeRobot commit metadata.
- Do not commit datasets, checkpoints, videos, caches, or raw simulator output.

## Required checks only

Before committing each phase, run only the checks relevant to that phase:

```bash
uv run ruff check
uv run pytest -q
```

Add the CUDA smoke test, packed numerical comparison, or LIBERO evaluation only when its corresponding phase requires it. Do not add a coverage target, repository-wide audit, security review, or unrelated refactor.

## Public interfaces

- `BitWAMConfig`: registered as policy type `bitwam`; exposes source checkpoint, world-loss weight, quantization scope, and inference backend.
- `BitWAMPolicy`: the LeRobot-compatible training and inference policy.
- `TernaryLinear.from_linear(...)`: converts an eligible BF16 linear layer without changing its tensor shapes.
- `convert_for_qat(policy, scope) -> QuantizationReport`: performs deterministic conversion and reports coverage.
- `bitwam train|evaluate|export|benchmark|summarize --config <yaml>`: the project command-line interface.

## Remote-agent handoff

Clone the renamed repository and work from the plan branch:

```bash
git clone git@github.com:adeeb10abbas/BitWAM.git
cd BitWAM
git switch ali/claude
uv sync
```

Read this file completely before editing. Begin with Phase 1, execute phases sequentially, and push each completed phase to `ali/claude`. Long GPU jobs should be resumable, skip completed outputs, and write a PID and log file so another agent can inspect or resume them.

V-JEPA2 remains BF16 and training-only because it is not part of deployed inference. V1 targets Linux with Ampere-or-newer NVIDIA GPUs; CPU and macOS training are out of scope.
