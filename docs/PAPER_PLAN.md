# BitWAM paper plan

## Working title and thesis

**BitWAM: World-Model-Assisted Post-Training for Ternary Vision-Language-Action
Policies**

The central question is whether action-conditioned future-state supervision can
improve or preserve closed-loop control when the policy backbone and predictive
world head use ternary matrix weights. BitWAM is not presented as a new ternary VLA
backbone. It uses released BitVLA weights as the controlled substrate and studies a
different problem: predictive post-training under extreme quantization.

No performance claim is valid until a saved BitWAM checkpoint completes closed-loop
LIBERO evaluation. Predictor loss is a representation diagnostic, not task success.

## Positioning

- BitNet supplies the absmean ternary-weight and per-token INT8-activation recipe.
- BitVLA supplies the native ternary VLA control and establishes that a 1.58-bit VLA
  can retain manipulation behavior.
- VLA-JEPA and V-JEPA 2-AC establish leakage-free latent future prediction and
  action-conditioned latent dynamics.
- LaWAM feeds predicted latent subgoals back into a policy at inference time.
- WLA-0 uses a World Expert whose prediction can be disabled at inference, so
  training-only prediction is prior art rather than a BitWAM novelty claim.
- BitWAM studies compression-aware auxiliary world modeling: a ternary predictor
  changes the policy through training gradients and is removed at inference, so the
  deployed controller has no additional world-model latency.

Avoid a “first 1-bit WAM” claim until the final related-work search is complete.

## Method

For current observation `o_t`, instruction `l`, proprioception `p_t`, and demonstrated
action chunk `a_t:t+H`, the native ternary VLA produces action-token hidden states
`h_t`. A stopped-gradient, fixed visual encoder maps the observation after the full
chunk to target `z_t+H`.

The BitWAM predictor uses three ternary linear matrices, absmean weight quantization,
per-token 8-bit activation quantization, and straight-through gradients:

`z_hat_t+H = W_phi(h_t, a_t:t+H)`

`L_world = 1 - cosine(z_hat_t+H, stopgrad(z_t+H))`

To prevent a static-scene shortcut, a within-batch negative pairs the same action
token state with another trajectory's action chunk. Let `z_hat^-_t+H` be the
prediction produced with the shuffled action. The action-conditioning objective is

`L_contrast = max(0, m - cosine(z_hat_t+H, z_t+H) + cosine(z_hat^-_t+H, z_t+H))`

`L_total = L_action + lambda * (L_world + beta * L_contrast)`

The primary run uses `m = 0.05` and `beta = 1`. We report the correct-minus-shuffled
cosine gap as a diagnostic; future-state cosine alone is insufficient evidence of
action-conditioned prediction.

The future frame is a target only. It never enters the student policy input. The
visual target space remains fixed during joint training.

Training has three explicit stages:

1. Fit a BF16 predictive head while the released controller is frozen.
2. Convert the warmed head to ternary forward passes and calibrate it while the
   controller remains frozen.
3. Jointly post-train action and world losses with a conservative policy learning
   rate; jointly optimize the shuffled-action margin; discard the predictor for
   deployed action inference.

## Required comparisons

| Run | Policy updates | Predictor | World weight | Purpose |
| --- | --- | --- | ---: | --- |
| Released BitVLA | none | none | 0 | Native ternary control |
| Action-only post-train | yes | compute-matched ternary branch | 0 | Isolate continued training |
| BitWAM BF16-head | yes | BF16 | 0.1 | Predictor-precision ablation |
| BitWAM ternary-head | yes | ternary | 0.1 | Primary method |

All post-training runs use the same examples, future-frame offset, action horizon,
global batch, optimizer schedule, initial controller checkpoint, and evaluation
initial states. Add a shuffled-action or state-only diagnostic before claiming that
the predictor learned action-dependent dynamics rather than scene similarity.

## Evaluation gates

1. Environment gate: one rollout on each of the 10 LIBERO-10 tasks.
2. Early selection: five rollouts per task for every serious checkpoint.
3. Paper result: 50 rollouts per task, three training seeds, with the same ordered
   initial states across methods.
4. Report mean success, per-task success, paired bootstrap confidence intervals,
   action L1, future-latent cosine similarity, wall-clock training cost, inference
   latency, peak GPU memory, and eligible ternary-weight storage.
5. The primary retention gate is at least 95% of the released native BitVLA success
   rate. A positive world-model claim additionally requires BitWAM to beat the
   action-only control across seeds.

## Evidence available now

- Post-hoc ternarization of the prior VLA-JEPA/Qwen policy failed all 10 recovery
  rollouts, despite finite and decreasing training losses.
- The released native ternary BitVLA control completed 10/10 LIBERO-10 smoke
  rollouts.
- Frozen-controller BF16 predictor pretraining increased future-latent cosine
  similarity from approximately 0 to above 0.90.
- The ternary predictor passes strict BF16-checkpoint loading and CUDA backward; all
  effective matrix weights are in `{-1, 0, +1}`.
- At 2,000 joint updates, ternary BitWAM completed 10/10 smoke rollouts and 45/50 in
  the paired seed-0 evaluation. Action-only scored 44/50 and BF16-head BitWAM 47/50.
- The ternary-versus-action-only difference is +2 points with paired bootstrap 95%
  interval `[-8,+12]`; this does not support an improvement claim.
- Two independent B200 profiles per policy find the same 5.433 GiB loaded
  allocation and 6.032 GiB query peak for all four action paths. Mean p50 latency is
  108.69–110.54 ms across methods; the unchanged graph supports a zero-overhead
  result, not a BitWAM speedup.
- A new exact packed runtime stores four ternary codes per byte. Text-backbone packing
  returns bit-identical actions, passes 10/10 ordered smoke rollouts, reduces resident
  CUDA allocation from 5.433 to 2.060 GiB (62.08%), and measures mean p50 107.82 ms
  versus 108.69 ms dense across two 100-query runs. Its mean p95 is worse, so this is
  a small median result rather than a broad latency claim.
- Full exact packing reduces resident allocation to 1.427 GiB (73.73%) and passes
  10/10, but raises p50 by 5.22%. Activation fusion reaches 106.93 ms p50 but changes
  actions and scores 9/10, so it is an ablation rather than the selected runtime.
- The standard loader still reads the dense 5.403-GiB artifact before packing; disk
  size and startup peak are not yet reduced. The training-only ternary head also
  retains 21.046 MiB of BF16 master parameters.

These observations establish that the method works in closed loop, but do not yet
establish a control benefit over action-only across training seeds.

## Primary references

- BitNet: <https://arxiv.org/abs/2402.17764>
- BitVLA: <https://arxiv.org/abs/2506.07530>
- V-JEPA 2: <https://arxiv.org/abs/2506.09985>
- VLA-JEPA: <https://arxiv.org/abs/2602.10098>
- LaWAM: <https://arxiv.org/abs/2606.15768>
- WLA-0: <https://arxiv.org/abs/2606.05979>
