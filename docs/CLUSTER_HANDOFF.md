# BitWAM cluster handoff

## The goal

Produce a BitWAM policy that actually completes LIBERO tasks after compression.
Training loss is supporting information. Closed-loop task success is the result that
decides whether a policy works.

The fastest useful cluster strategy is:

1. prove the cluster setup with one training update and ten short rollouts;
2. run several independent, bounded policy variants in parallel;
3. evaluate every serious candidate early;
4. promote only candidates that move toward the task correctly;
5. stop a failed compression path instead of tuning it indefinitely.

Do not add Kubernetes, a new training framework, a dashboard, or a general experiment
manager. Use the cluster's existing allocation method and one YAML file per run.

This handoff reflects the user's later request to use cluster scale for more variety.
It replaces the execution plan's strict one-candidate pilot sequence only with the
seven bounded training jobs listed below. It does not authorize an open-ended sweep,
weaker success gate, early Qwen+DiT run, or premature packed-inference work.

## Current state in plain language

### What works

- The upstream BF16 VLA-JEPA policy completed **49 of 50** LIBERO-10 episodes.
- BitWAM's BF16 wrapper matches upstream actions in parity tests.
- CUDA forward/backward tests pass.
- Qwen ternary training runs across two local RTX 3090s without numerical errors.
- The repository test suite passes: **29 tests passed**.

### What does not work yet

- The Qwen-ternary checkpoint saved at training step 1,000 completed **0 of 50**
  LIBERO-10 episodes.
- All ten tasks failed, five attempts per task.
- The checkpoint loaded correctly. The rollouts showed incorrect robot motion, so this
  was a policy failure rather than a simulator or checkpoint-loading failure.
- The original process later reached step 1,667, but only step 1,000 was saved. The
  later unsaved updates were discarded when training was stopped for evaluation.
- No compressed BitWAM policy is currently proven to work.

### Important evidence boundary

The 49/50 baseline used the upstream VLA-JEPA checkpoint directly. It did not run a
saved local BitWAM BF16 checkpoint through all 50 episodes. The wrapper parity tests
are strong evidence, but a cluster run should still produce that closed-loop BitWAM
BF16 control.

### Recovery currently running locally

The local workstation is running the one predefined Qwen recovery:

- first and last Qwen blocks remain BF16;
- all eligible middle Qwen attention and MLP layers are ternary;
- learning rate is `2.5e-5`;
- world loss weight is `0.1`;
- 2,000 updates, saving at steps 1,000 and 2,000;
- two RTX 3090s, effective batch size 8;
- no Kubernetes.

At this handoff snapshot it had passed step 100 with no restart, out-of-memory error,
NaN, or hardware thermal slowdown. Recent averaged losses were:

| Step | Total loss | Action loss | World loss |
| ---: | ---: | ---: | ---: |
| 20 | 0.979 | 0.846 | 0.133 |
| 40 | 0.669 | 0.538 | 0.131 |
| 60 | 0.437 | 0.309 | 0.128 |
| 80 | 0.374 | 0.249 | 0.126 |
| 100 | 0.377 | 0.252 | 0.124 |

This is stable training, not evidence of a functioning policy. The failed run also
reached a low training loss while producing 0/50 task success.

Leave the local recovery running unless the local GPUs are needed for something else.
It provides one independent result while the cluster starts faster runs.

## Repository and useful files

Clone and use the existing branch:

```bash
git clone https://github.com/adeeb10abbas/BitWAM.git
cd BitWAM
git switch ali/claude
```

Read these files in this order:

1. `docs/CLUSTER_HANDOFF.md`
2. `docs/EXECUTION_PLAN.md`
3. `docs/STATUS.md`
4. `configs/pilot_qwen_recovery.yaml`

The compact results are in:

- `results/baseline-bf16-seed0/`
- `results/pilot-qwen-seed0/`

Raw checkpoints, simulator logs, and videos are intentionally outside Git.

## Use the cluster for breadth, not unnecessary distribution

The model has about 2.77 billion trainable parameters. Large-memory GPUs should run
one complete experiment per GPU. This produces more independent answers than using
many GPUs to make one small 2,000-step job finish slightly sooner.

Recommended assignment:

| GPU type | Best BitWAM use |
| --- | --- |
| B200, 192 GB | Highest-priority training. Start with one GPU per run. |
| RTX PRO 6000 Blackwell, 96 GB | One training run per GPU; also good for overflow evaluation. |
| RTX A6000 or A40, 48 GB | One evaluation per GPU. Use two GPUs per training run if a single-GPU smoke test does not fit comfortably. |
| 24 GB cards | Use the proven two-GPU sharded recipe. |

If “RTX 6000” means a 48 GB Ada/Ampere model rather than the 96 GB RTX PRO 6000
Blackwell model, treat it like the A6000/A40 row. Check the exact name and memory with
`nvidia-smi` before assigning jobs.

Memory references: [B200](https://docs.nvidia.com/brev/reference/gpu-types),
[RTX PRO 6000 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/),
[RTX A6000](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20%281%29.pdf),
and [A40](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a40/NVIDIA%20A40%20Product%20Brief.pdf).

For a B200 or 96 GB RTX PRO 6000, begin with:

- `num_processes: 1`
- `batch_size: 8`
- `gradient_accumulation_steps: 1`
- `fsdp_cpu_offload: false`

That preserves the pilot's effective batch size of 8 without cross-GPU communication.
If the one-update smoke test does not fit, use two GPUs and the existing local recipe:

- `num_processes: 2`
- `batch_size: 4`
- `gradient_accumulation_steps: 1`
- `fsdp_cpu_offload: true`

Do not spend time tuning distributed training before the one-GPU smoke test proves it
is necessary.

## First 30 minutes on the cluster

### 1. Confirm the environment

```bash
nvidia-smi
uv sync --frozen
uv run bitwam --help
uv run pytest -q
```

Expected result: 29 tests pass.

Use fast local storage for the shared Hugging Face cache when available. Download the
dataset and model once per node rather than once per experiment. Keep the current
streaming dataset mode for the first real run; changing the data pipeline is not part
of this handoff.

### 2. Run one real update on the target GPU layout

Copy `configs/smoke_qat_dual_gpu.yaml` to a cluster-specific YAML. Change only the
GPU count, batch size, output path, and run name as described above. Run:

```bash
uv run bitwam train --config configs/cluster-smoke.yaml
```

Pass conditions:

- one optimizer update completes;
- total, action, and world losses are finite;
- the process exits successfully;
- the selected GPUs show real compute activity.

Do not call an import test or model load a training success.

### 3. Run a ten-episode rollout check

Use ten episodes across LIBERO-10, which gives one episode per task. This is only an
environment check, not a publishable score.

Pass conditions:

- all ten task environments start;
- ten nonempty videos are written;
- the evaluator writes its result JSON and exits successfully.

If setup fails, fix only the direct blocker. Do not start an environment rewrite.

## Fast experiment program

The main unknown is not the perfect learning rate. It is **how much of Qwen can be
ternary before the robot loses task-directed behavior**. Spend the cluster budget on
that question first.

### Wave 1: no new model design

Start these immediately and independently:

| Candidate | Change | Seeds | Purpose |
| --- | --- | ---: | --- |
| BF16 BitWAM control | No ternary layers | 0 | Prove the wrapper itself works in closed loop. |
| Full-Qwen reference | Existing full-Qwen scope, `1e-4` LR | 0 | Complete the previously interrupted reference to 2,000 steps. |
| Edge-BF16 recovery | Existing recovery, `2.5e-5` LR | 0, 1, 2 | Test whether the predefined recovery is real or seed-specific. |

Run seed 0 of the edge-BF16 recovery on a B200 even though it is already running
locally. The cluster copy will probably finish earlier, while the local copy remains an
independent repeat.

For the BF16 BitWAM control, use the existing trainer to materialize the wrapper with
one update, `save_freq: 1`, and `learning_rate: 0.0`, then evaluate that saved local
checkpoint. Confirm that its weights remain unchanged. If the pinned scheduler rejects
a zero learning rate, add the smallest direct materialization helper; do not build the
Phase 5 packing/export system just to create this control.

### Wave 2: three small boundary probes

If enough GPUs are available, add exactly these three fixed variants. This is the only
recommended model-code extension before rollout evidence:

| Candidate | Ternary layers | BF16 layers | Question answered |
| --- | --- | --- | --- |
| Attention only | Qwen attention projections | Qwen MLPs and protected layers | Is attention ternarization the failure source? |
| MLP only | Qwen MLP projections | Qwen attention and protected layers | Is MLP ternarization the failure source? |
| Middle half | Attention and MLP in the middle half of Qwen blocks | Outer quarter of blocks and protected layers | Does a less aggressive contiguous boundary retain behavior? |

Keep all three at:

- seed 0;
- learning rate `2.5e-5`;
- world loss weight `0.1`;
- effective batch size 8;
- 2,000 updates.

Implementation must stay small: add three named, deterministic selectors to the
existing conversion function and focused boundary tests. Do not build a generic layer
selection language or configuration framework.

This produces seven useful training jobs in the first wave: one full-Qwen reference,
three edge-recovery seeds, and three boundary probes. The BF16 control is primarily an
evaluation job.

### Why not start Qwen+DiT yet?

Qwen-only compression currently scores 0/50. Adding DiT compression before any Qwen
candidate works makes the failure harder to interpret. Run Qwen+DiT only after at least
one Qwen candidate passes the policy gate.

### Why not run a large learning-rate sweep?

The current evidence points to compression boundary as the main unknown. A broad
learning-rate grid would create many low-value runs with the same likely failure mode.
Use the two already meaningful rates: `1e-4` for the original reference and `2.5e-5`
for recovery/boundary candidates.

## Evaluation schedule

### At step 1,000: fast screen

As soon as each step-1,000 checkpoint is complete, evaluate **10 episodes**, one from
each LIBERO-10 task, on a separate A40/A6000-class GPU. Training can continue toward
step 2,000 while evaluation reads the fixed step-1,000 checkpoint.

Record:

- successes out of 10;
- which tasks succeeded;
- whether the robot moved toward the correct object;
- obvious repeated failure behavior;
- evaluation time.

The video check matters. A 0/10 result with task-directed reaching is different from a
0/10 result where the arm moves to the same wrong location every time.

### At step 2,000: real gate

Run **50 episodes** for every candidate that either:

- succeeded at least once in the ten-episode screen; or
- showed clearly task-directed motion and continued improving.

Run each candidate evaluation on its own GPU in parallel. Do not split a single
candidate's score across custom scripts unless the stock evaluator becomes a proven
bottleneck.

The Phase 4 gate is **45 successes from 50 episodes**. Training loss cannot substitute
for this score.

### Promotion rule

- **45/50 or better:** candidate passes. Repeat the best boundary on seeds 1 and 2 if
  those repeats do not already exist.
- **1–44/50:** candidate is partially functional. Keep the result, compare videos, and
  promote at most the best two candidates for one predefined follow-up.
- **0/50 with incorrect motion:** stop that candidate.
- **No Qwen candidate reaches 45/50:** stop expanding ternary compression. Produce a
  functioning BF16 BitWAM checkpoint and report Qwen ternarization as a negative result.

Do not quietly lower the gate after seeing results.

## After a Qwen candidate passes

Use only the best Qwen boundary for Qwen+DiT:

1. run one seed of Qwen+DiT for 2,000 updates;
2. screen ten episodes at step 1,000;
3. evaluate 50 episodes at step 2,000;
4. if needed, use the existing one-time recovery that keeps the last four DiT blocks
   BF16;
5. stop if that recovery misses 45/50.

Do not implement packed inference until a compressed policy passes closed-loop
evaluation. A smaller model that cannot complete tasks is not a useful deployment.

## Final scalable experiment

Only after the Qwen+DiT gate passes, run the planned 12-job matrix:

- BF16 with world loss;
- BF16 without world loss;
- ternary with world loss;
- ternary without world loss;
- seeds 0, 1, and 2 for each condition.

Run the 12 jobs independently across 12 large-memory GPUs when possible. This is more
robust and simpler than one multi-node job.

Preserve effective batch size 64:

- one large-memory GPU: microbatch 8, accumulation 8;
- two-GPU 48 GB job: microbatch 4 per GPU, accumulation 8.

Evaluate every final checkpoint on all ten LIBERO tasks with 50 episodes per task, as
required by the execution plan. Keep task-level scores separate.

## Minimal reporting format

Create one Markdown report per serious candidate under `results/<candidate>/REPORT.md`.
Use plain language and this structure:

```markdown
# <candidate name>

## What we changed
One paragraph describing the layers kept BF16 and the layers made ternary.

## Training
- GPU type and count:
- Updates completed:
- Wall-clock time:
- Final action loss:
- Final world loss:
- Any crash, restart, or numerical problem:

## Policy evaluation
- Successes / episodes:
- Per-task successes:
- Evaluation time:
- What the videos show:

## Decision
Pass, partial, or stop. State why in one paragraph.

## Next action
The single next experiment this result authorizes.
```

Do not fill the report with commit hashes, checkpoint hashes, internal object names,
long command dumps, or raw logs. The YAML config and saved artifacts already provide
the technical detail. Reports should answer: what changed, did the robot work, what
failed, and what happens next.

Keep machine-readable compact metrics beside the report. Keep checkpoints, videos,
datasets, and raw logs outside Git.

## Progress updates

Send an update only at useful milestones:

1. cluster setup and smoke test completed;
2. each step-1,000 screen completed;
3. each 50-episode result completed;
4. any failed job that needs a decision;
5. final promotion or stop decision.

Each update should lead with the policy result. Example:

> MLP-only BitWAM completed 37/50 tasks. It is functional but below the 45/50 gate.
> Failures concentrate in tasks 7–9, where the arm reaches the correct object but does
> not finish the placement. We are promoting attention-only and middle-half, not adding
> another learning-rate sweep.

## Ready-to-paste prompt for the cluster Codex task

```text
Clone https://github.com/adeeb10abbas/BitWAM.git and switch to ali/claude.
Read docs/CLUSTER_HANDOFF.md completely, then docs/EXECUTION_PLAN.md and docs/STATUS.md.

Goal: produce a closed-loop functioning BitWAM policy quickly. Use the cluster's GPUs
for independent experiments, not Kubernetes and not unnecessary multi-node training.
Start with the 30-minute setup gate in the handoff. Prefer one B200 or one 96 GB RTX PRO
6000 per training run; use A40/A6000-class GPUs for parallel evaluations. Run Wave 1
immediately, then the three bounded boundary probes if resources permit. Evaluate ten
episodes at step 1,000 and 50 episodes at step 2,000. The pass gate is 45/50.

Do not treat low training loss as policy success. Do not start Qwen+DiT or packed
inference until a Qwen candidate passes. Keep code changes limited to the three fixed
boundary selectors and focused tests. Write plain-language REPORT.md files that say
what changed, how many tasks succeeded, what videos show, and the next decision. Do not
fill progress reports with hashes or internal jargon. Keep me posted at meaningful
milestones.
```

## Immediate next action

On the fastest available B200:

1. clone the branch;
2. run the tests and one-update smoke;
3. launch edge-BF16 recovery seed 0 as a single-GPU effective-batch-8 run;
4. launch seeds 1 and 2 on two additional large-memory GPUs;
5. use separate A40/A6000 GPUs for the step-1,000 screens as checkpoints appear.

In parallel, prepare the BF16 BitWAM control and the three small boundary probes. This
gets to closed-loop policy numbers quickly while keeping the experiment set bounded and
interpretable.
