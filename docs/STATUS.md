# Execution status

Status recorded on 2026-08-15 from branch `ali/claude`.

## Active native BitVLA track

The paper track now uses the released native ternary BitVLA controller at upstream
revision `8afac0260b3748b14657a69ec58e3d9f0d6da3a7`. This replaces the unsuccessful
post-hoc Qwen ternarization route as the primary substrate; the older experiments
remain useful negative evidence.

Verified closed-loop LIBERO-10 smoke results, one ordered rollout per task:

- Released native BitVLA at step 100,000: 10/10.
- Matched action-only continued training at step 101,000: 10/10.
- Matched action-only continued training at step 102,000: 10/10.

The world-model training path has also passed these gates:

- Frozen-controller BF16 predictor pretraining increased future-latent cosine above
  0.92 while leaving saved action and proprioception tensors bit-identical to the
  released controller.
- A ternary predictor loaded the BF16 state strictly, used only effective matrix
  levels `{-1, 0, +1}`, completed CUDA backward, and recovered approximately 0.94
  future-latent cosine during calibration.
- The pre-contrastive correct-versus-shuffled action gap remained only about
  `5e-4`, exposing a static-scene shortcut. The joint objective now includes a 0.05
  shuffled-action margin.

Two 2,000-update joint runs are active on the four-B200 Kubernetes pod: the primary
ternary predictor on GPUs 1/3 and the BF16 predictor ablation on GPUs 0/2. Each has
an automatic 10-task evaluator waiting on its final saved checkpoint. The cgroup OOM
kill counter has not increased during these jobs.

The current local check result is `60 passed, 1 skipped`; the skip is CUDA-only on
the macOS host. The Python linter passes.

## Historical local VLA-JEPA track

- Phase 1 acceptance: clean Python 3.12 package, exact `uv.lock`, import,
  `bitwam --help`, and CPU tests.
- Phase 2 acceptance: registered wrapper, upstream processor reuse, native
  source loading, exact seeded CPU action parity, CPU forward/backward, and
  CUDA forward/backward on each local RTX 3090.
- Phase 3 acceptance: BF16 master weights, ternary/INT8 simulation, STE update,
  deterministic conversion boundaries, recovery boundaries, save/load, and
  quantization coverage reporting.
- Phase 4 BF16 baseline: 49 successes from 50 closed-loop LIBERO-10 episodes
  (98%). Nine tasks scored 5/5 and task 8 scored 4/5. All 50 viewport videos,
  the raw evaluator JSON, and the exact command/log/state are present under
  `outputs/baseline-bf16-seed0/`.
- Phase 4 QAT smoke: one real Qwen-ternary training step completed across both
  local RTX 3090s with two-process FSDP, BF16 mixed precision, streaming data,
  CPU state offload, and the original gradient clipping threshold. The finite
  effective global batch 8. The finite metrics were loss `0.822`, action loss
  `0.688`, world loss `0.135`, and gradient norm `179.853`; the optimizer update
  completed and the process exited zero. The reported peak allocation was
  12.72 GiB per rank.

The check result at that point was `29 passed, 1 warning`. The warning is the
expected CPU-autocast warning in a CPU-only delegation test; both CUDA acceptance
tests pass.

## Local hardware recipe

Training uses the two installed 24 GiB RTX 3090s directly. The pilot launcher
starts two local workers and shards the 2.77B-parameter policy with FSDP. It does
not use Kubernetes or any remote scheduler.

The legacy `HuggingFaceVLA/libero` Parquet dataset expands beyond local disk
capacity with the map-style loader, so the configs use LeRobot's streaming
loader. The pinned LeRobot/Accelerate combination needs two narrow compatibility
fixes supplied by `lerobot_policy_bitwam.train_entrypoint`: independent iterable
batches preserve language strings, and CPU-offloaded gradient shards reduce
their norm through a GPU scalar collective. Both behaviors have focused tests.
Each GPU rank uses one loader worker; using four per rank replicated the
streaming buffers until a host-memory kill, while the one-worker recipe completed
the real optimizer-step acceptance run.

The 2,000-step pilot keeps the original effective batch size of 8 as
`4 samples × 2 GPU workers`, with no gradient accumulation. LeRobot counts
microbatches as steps, so this layout preserves all 2,000 optimizer updates. It
saves at steps 1,000 and 2,000; a failed job automatically resumes from
`checkpoints/last`.

## Historical Phase 4 gate

The Qwen pilot must score at least 45 successes from 50 episodes to reach 90% of
the measured BF16 result. Only then may the `qwen_dit` pilot run. The original
execution plan limits each recovery to one attempt; the later cluster handoff adds
only the bounded seed repeats and three fixed Qwen-boundary probes listed there.

The saved step-1,000 full-Qwen checkpoint scored 0/50. It loaded correctly and
produced incorrect closed-loop motion on all ten tasks. The original training
process was stopped later at an unsaved step 1,667 so that evaluation could run.

The predefined edge-BF16/lower-learning-rate Qwen recovery is now running locally
on both RTX 3090s. Its training is numerically stable, but it has not yet produced
a closed-loop result. Training loss must not be reported as policy success.

Compact baseline provenance and metrics are committed under
`results/baseline-bf16-seed0/`. The failed Qwen evaluation is summarized under
`results/pilot-qwen-seed0/`; raw simulator artifacts remain ignored.

For the cluster continuation, GPU assignment, bounded experiment variants,
evaluation schedule, stop rules, and plain-language report format, see
`docs/CLUSTER_HANDOFF.md`.

## Remaining work

- Finish and evaluate the active BF16- and ternary-predictor joint checkpoints.
- Promote only checkpoints that retain at least 95% of released control success.
- Run 50 rollouts per task for three seeds with paired initial states.
- Measure packed ternary storage, peak VRAM, training cost, and deployed inference
  latency.
- Replace pending paper cells with archived multi-seed metrics and confidence
  intervals.

No quantized task-success, compression, or latency claim is authorized until
the corresponding closed-loop or packed-inference artifact exists.
