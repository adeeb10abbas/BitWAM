# Full-Qwen ternary pilot

## What we changed

Eligible Qwen attention and MLP linear layers used ternary weights during training.
The vision encoder, world model, action model, embeddings, normalization layers, and
final action output remained BF16.

## Training

- Two RTX 3090 GPUs were used directly, without Kubernetes.
- The run was configured for 2,000 updates with an effective batch size of 8.
- A complete checkpoint was saved at step 1,000.
- Training later reached step 1,667 and remained numerically stable.
- The user requested immediate evaluation, so training was stopped. Updates after
  step 1,000 had not been saved and were not evaluated.

The falling training loss looked encouraging, but it did not prove that the robot
could complete tasks.

## Policy evaluation

- **0 successes from 50 episodes** across LIBERO-10.
- Every task scored 0/5.
- All 50 rollout videos were written successfully.
- The checkpoint loaded without missing model weights.
- Video inspection showed the arm moving to incorrect locations instead of completing
  the requested tasks.

## Decision

This checkpoint failed the 45/50 gate by a wide margin. It is not a functioning
compressed policy. The result supports testing less aggressive Qwen boundaries rather
than starting Qwen+DiT compression or a broad learning-rate sweep.

## Next action

The edge-BF16/lower-learning-rate recovery is running locally. The cluster should run
that recovery and the bounded attention-only, MLP-only, and middle-half probes described
in `docs/CLUSTER_HANDOFF.md`, with closed-loop evaluation deciding promotion.
