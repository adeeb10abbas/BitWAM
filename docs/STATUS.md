# Execution status

Status recorded on 2026-08-14 from branch `ali/claude`.

## Verified locally

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

The current required check result is `29 passed, 1 warning`. The warning is the
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

## Active Phase 4 gate

The Qwen pilot must score at least 45 successes from 50 episodes to reach 90% of
the measured BF16 result. Only then may the `qwen_dit` pilot run. Each predefined
recovery config remains limited to one attempt exactly as stated in the plan.

Compact baseline provenance and metrics are committed under
`results/baseline-bf16-seed0/`; raw simulator artifacts remain ignored.

## Remaining work

- Complete and evaluate the 2,000-step Qwen pilot.
- Run the gated Qwen+DiT pilot, using a recovery only if its primary run misses.
- Implement and validate genuinely packed reference and Triton inference.
- Run and evaluate the gated 12-run final matrix.
- Package final manifests, aggregate metrics, size/VRAM/latency results, and the
  compact results table.

No quantized task-success, compression, or latency claim is authorized until
the corresponding closed-loop or packed-inference artifact exists.
