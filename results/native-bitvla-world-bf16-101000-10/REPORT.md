# BF16-head BitWAM at 1,000 joint updates

The predictor-precision ablation completed all 10 ordered LIBERO-10 smoke rollouts,
one trial per task. Its correct-versus-shuffled action-conditioning gap was 0.2239,
correct-target cosine was 0.9414, and action L1 was 0.0329 at the checkpoint.

This result shows that the shared predictive objective is compatible with retained
control in BF16. Together with the ternary-head 10/10 result, it supports evaluating
both full-budget checkpoints. It does not show that either predictor improves control
over the 10/10 action-only comparison.
