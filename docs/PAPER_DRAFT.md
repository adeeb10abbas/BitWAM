# BitWAM: Predictive Post-Training for Ternary Vision-Language-Action Policies

> Living draft. Results are filled only from archived artifacts. The completed
> seed-0 evaluation has five rollouts per task; the preregistered multi-seed paper
> matrix remains pending.

## Abstract

Native ternary vision-language-action (VLA) policies can retain closed-loop robot
control, but it is not yet clear whether action-conditioned predictive supervision
remains useful when both control and prediction operate under extreme weight
quantization. We present BitWAM, a staged post-training method that adds a latent
future-state predictor to a released BitVLA controller. A frozen-controller BF16
stage first learns the predictive task, a calibration stage converts the predictor
to absmean ternary matrix weights with per-token INT8 activations, and a final stage
jointly optimizes action regression and prediction. A shuffled-action margin prevents
the predictor from satisfying the objective through static scene similarity. The
predictor is used only during training and adds no deployed policy latency. In a
paired seed-0 LIBERO-10 evaluation with five rollouts per task, released BitVLA,
matched action-only post-training, ternary-head BitWAM, and BF16-head BitWAM achieve
39/50, 44/50, 45/50, and 47/50 success. Ternary BitWAM meets the predeclared 45/50
gate and raises the correct-versus-shuffled cosine gap from approximately `5e-4` to
0.215. Its two-point advantage over action-only is unresolved, however (paired
bootstrap 95% interval `[-8,+12]` points; exact McNemar p=1.0). Additional training
seeds are required for a control-improvement claim.

## 1. Introduction

Robot policies must combine semantic understanding with precise, temporally
coherent control under tight deployment budgets. Vision-language-action models
provide a flexible interface for this problem, while BitVLA demonstrates that a
native 1.58-bit VLA can preserve manipulation ability. Compression alone, however,
does not tell us how best to adapt such a controller. Ordinary action-only
fine-tuning provides supervision at the output, but does not explicitly require the
policy representation to encode how an action chunk changes the scene.

Latent predictive objectives offer a natural auxiliary signal: from a current
observation and action chunk, predict the visual representation after executing the
chunk. Applying this idea to a ternary controller is not a routine combination.
Post-hoc quantization can destroy a pretrained VLA even when its training losses are
finite, and a future-frame loss can be minimized by predicting persistent scene
content while ignoring the action. BitWAM addresses both problems with native
ternary training and an explicit action-conditioning test.

Our aim is deliberately narrower than proposing another ternary VLA backbone.
BitVLA supplies the controller and released checkpoint. We investigate how to add
predictive post-training under the same extreme-quantization constraint while
preserving closed-loop behavior and zero auxiliary-model inference overhead.

The paper makes three empirical and methodological contributions:

1. A three-stage recipe—BF16 predictor pretraining, ternary calibration, and joint
   post-training—that preserves the released native ternary controller rather than
   applying destructive post-hoc backbone quantization.
2. An action-contrastive latent objective and a correct-versus-shuffled diagnostic
   that expose the static-scene shortcut hidden by high future-latent cosine.
3. A matched evaluation against released control, continued action-only training,
   and a BF16 predictor ablation, with the predictive branch removed at deployment.

## 2. Related work

### Extreme-weight quantization

BitNet introduces absmean ternary weight quantization and per-token activation
quantization for 1.58-bit language models. BitVLA transfers native ternary training
to vision-language-action control and provides the substrate used in this work.
BitWAM does not claim either backbone contribution; it studies predictive
post-training of that substrate.

### Latent predictive models for control

V-JEPA 2 and its action-conditioned variant motivate prediction in a learned visual
representation rather than pixel reconstruction. VLA-JEPA applies joint-embedding
prediction to VLA learning. LaWAM uses predicted latent subgoals in a world-action
model. WLA-0 uses an auxiliary World Expert whose prediction can be disabled during
inference or enabled for test-time scaling. Consequently, training-only prediction
is not itself our novelty. BitWAM instead isolates compression-aware predictive
post-training of a released native ternary controller: staged BF16-to-ternary
predictor calibration, a ternary auxiliary head, and an explicit shuffled-action
dependence objective, with the deployed action path unchanged.

## 3. Method

### 3.1 Released ternary controller

Let the current observation be `o_t`, language instruction `l`, proprioceptive state
`p_t`, and demonstrated action chunk `a_t:t+H`. The released BitVLA policy produces
an action prediction and action-token hidden states `h_t`. The policy action loss is

`L_action = ||a_hat_t:t+H - a_t:t+H||_1`.

We preserve BitVLA's native ternary parameterization. The failed post-hoc
ternarization run is evidence about an unsuitable conversion procedure, not evidence
that ternary VLA control is impossible.

### 3.2 Leakage-free latent target

The current observation is the only visual input to the student policy. A fixed
visual encoder maps the observation after the complete demonstrated chunk to a
target representation:

`z_t+H = stopgrad(E(o_t+H))`.

The future frame is never concatenated to the policy input. Freezing the target
encoder during joint training prevents the target space from moving to satisfy the
auxiliary objective.

### 3.3 Ternary predictor

The predictor pools the action-token hidden states, embeds the continuous action
chunk, and applies a three-matrix residual MLP:

`z_hat_t+H = W_phi(h_t, a_t:t+H)`.

For the primary model, every predictor matrix uses absmean ternary weights in
`{-alpha, 0, +alpha}`, per-token INT8 activation quantization, and a straight-through
gradient estimator. The head has 11,034,368 trainable parameters, of which
11,024,384 (99.91%) belong to the three ternary matrices. Those eligible matrices
require 2.76 MB under a simple two-bit packing or a theoretical 2.18 MB under
entropy-optimal ternary coding; the current training checkpoint intentionally retains
BF16 master weights and is not evidence of a packed runtime. The base prediction loss
is

`L_world = 1 - cos(z_hat_t+H, z_t+H)`.

### 3.4 Action-contrastive objective

High future-latent cosine does not establish action-conditioned prediction when much
of the scene remains unchanged. We construct an in-batch negative by pairing the
same hidden state with another example's action chunk. With negative prediction
`z_hat^-_t+H`, the margin loss is

`L_contrast = mean[max(0, m - cos(z_hat_t+H, z_t+H) + cos(z_hat^-_t+H, z_t+H))]`.

The optimized objective is

`L_total = L_action + lambda (L_world + beta L_contrast)`.

The primary configuration uses `lambda=0.1`, `beta=1`, and `m=0.05`. We separately
report the action-conditioning gap

`Delta_action = cos(z_hat_t+H, z_t+H) - cos(z_hat^-_t+H, z_t+H)`.

### 3.5 Staged training and deployment

Training proceeds in three stages. First, the controller and target encoder are
frozen while a BF16 predictor learns the future-latent objective. Second, the warmed
predictor weights are loaded into ternary-forward layers and calibrated with the
controller still frozen. Third, the controller and predictor are updated jointly at
a conservative policy learning rate while the target encoder remains fixed. The
predictor is discarded for evaluation and deployment; inference executes the native
BitVLA action path only.

## 4. Experimental protocol

We use the LIBERO-10 dataset and official closed-loop evaluator through the released
BitVLA code path. All post-training comparisons begin from the same released
checkpoint and use the same examples, image augmentation, future offset, action
horizon, global batch of 64, 2,000 update budget, learning-rate schedule, and ordered
evaluation initial states. Training uses two NVIDIA B200 GPUs per run.

The comparison matrix is:

| Method | Policy updates | Predictive head | Auxiliary weight | Purpose |
| --- | ---: | --- | ---: | --- |
| Released BitVLA | 0 | none | 0 | Native ternary reference |
| Action-only | 2,000 | compute-matched, unused | 0 | Continued-training control |
| BitWAM-BF16 | 2,000 | BF16 | 0.1 | Predictor-precision ablation |
| BitWAM-Ternary | 2,000 | ternary | 0.1 | Primary method |

The engineering screen runs one rollout on each task. The paper evaluation must run
50 rollouts per task for three training seeds with identical initial-state ordering.
We will report mean and per-task success, paired bootstrap confidence intervals,
action L1, future cosine, action-conditioning gap, wall-clock cost, peak GPU memory,
inference latency, and predictor storage.

## 5. Results

### 5.1 Closed-loop screen

| Method | Checkpoint | Successes / episodes | Status |
| --- | ---: | ---: | --- |
| Released BitVLA | 100,000 | 10 / 10 | verified |
| Action-only | 102,000 | 10 / 10 | verified |
| BitWAM-BF16 | 101,000 | 10 / 10 | verified halfway |
| BitWAM-Ternary | 101,000 | 10 / 10 | verified halfway |
| BitWAM-BF16 | 102,000 | 10 / 10 | verified final |
| BitWAM-Ternary | 102,000 | 10 / 10 | verified final |

The action-only result shows that the conservative continued-training recipe itself
retains closed-loop behavior. Consequently, matching released BitVLA is a retention
result, not evidence that the world objective improves control. A positive BitWAM
claim requires a paired advantage over action-only across the full multi-seed
evaluation. At 1,000 joint updates, both predictor precisions also retain 10/10 smoke
success; the ternary predictor has future cosine 0.9390 and an action-conditioning gap
of 0.1842, while BF16 has cosine 0.9414 and a gap of 0.2239.

### 5.2 Paired seed-0 evaluation

The next tier uses five rollouts per task with the same ordered initial states:

| Method | Successes / episodes | Rate | Difference vs. action-only |
| --- | ---: | ---: | ---: |
| Released BitVLA | 39 / 50 | 78% | -10 points |
| Action-only | 44 / 50 | 88% | reference |
| BitWAM-Ternary | 45 / 50 | 90% | +2 points |
| BitWAM-BF16 | 47 / 50 | 94% | +6 points |

Ternary BitWAM succeeds on four paired states where action-only fails and fails on
three where action-only succeeds. Its +2-point difference has paired bootstrap 95%
interval `[-8,+12]` and exact McNemar p=1.0. BF16 versus action-only is +6 points
with interval `[-2,+14]` and p=0.375. BF16 versus ternary is +4 points with interval
`[-4,+12]` and p=0.625. Thus this seed establishes a functioning ternary BitWAM and
passes the 45/50 gate, but does not resolve a control benefit from the world objective
or a BF16-versus-ternary precision difference.

### 5.3 Predictor diagnostics

The frozen-controller BF16 pretraining stage increases future-latent cosine above
0.92, and ternary calibration recovers approximately 0.94 in the initial run.
However, before contrastive post-training the correct-versus-shuffled cosine gap is
only about `5e-4`. This finding demonstrates why future cosine alone is misleading:
the predictor can model persistent visual content with minimal action dependence.
At 2,000 joint updates, ternary BitWAM reaches future cosine 0.9445 and action gap
0.2146; BF16 reaches 0.9486 and 0.2449. Both complete 10/10 smoke rollouts, and the
ternary method completes 45/50 in the seed-0 paired evaluation.

## 6. Limitations and claim boundaries

The current evidence is limited to simulation, one training seed, and five rollouts
per task. The auxiliary objective predicts a visual representation, not full physical
state, and the shuffled-action negative tests dependence rather than causal
correctness. Results on one released controller do not establish generality across
VLA families. We therefore avoid “first 1-bit world-action model” and control
improvement claims unless the related-work audit and paired multi-seed results
support them.

## References

- Wang et al. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.*
  <https://arxiv.org/abs/2402.17764>
- *BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation.*
  <https://arxiv.org/abs/2506.07530>
- Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding,
  Prediction and Planning.* <https://arxiv.org/abs/2506.09985>
- *VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model.*
  <https://arxiv.org/abs/2602.10098>
- *LaWAM: A Latent World Action Model for Vision-Language-Action Control.*
  <https://arxiv.org/abs/2606.15768>
- *World-Language-Action Model for Unified World Modeling, Language Reasoning, and
  Action Synthesis.* <https://arxiv.org/abs/2606.05979>
