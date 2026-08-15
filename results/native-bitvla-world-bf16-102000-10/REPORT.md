# BF16-head BitWAM at 2,000 joint updates

The full-budget predictor-precision ablation completed all 10 ordered LIBERO-10
smoke rollouts, one trial per task. Correct-target cosine reached 0.9486, the
correct-versus-shuffled action-conditioning gap reached 0.2449, and action L1 was
0.0313.

The BF16 head has slightly stronger predictor diagnostics than the ternary head in
this seed, while both retain 10/10 smoke success. The paired 50-rollout matrix is
required before interpreting any task-level difference.
