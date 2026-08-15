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
- Frozen-controller BF16 predictor pretraining is active on the cluster and has
  increased future-latent cosine similarity from approximately 0 to above 0.90.
- The ternary predictor passes strict BF16-checkpoint loading and CUDA backward; all
  effective matrix weights are in `{-1, 0, +1}`.

These observations motivate the method but do not yet establish its closed-loop
benefit.

## Primary references

- BitNet: <https://arxiv.org/abs/2402.17764>
- BitVLA: <https://arxiv.org/abs/2506.07530>
- V-JEPA 2: <https://arxiv.org/abs/2506.09985>
- VLA-JEPA: <https://arxiv.org/abs/2602.10098>
- LaWAM: <https://arxiv.org/abs/2606.15768>
