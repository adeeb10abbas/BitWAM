# Ternary-head BitWAM at 1,000 joint updates

The primary BitWAM checkpoint completed all 10 ordered LIBERO-10 smoke rollouts,
one trial per task. Its correct-versus-shuffled action-conditioning gap was 0.1842,
compared with roughly `5e-4` before contrastive post-training, while correct-target
cosine remained 0.9390 and action L1 was 0.0329.

The first evaluator launch encountered an NFS settlement race while reading a JSON
file. The checkpoint file subsequently validated, training had resumed, and the same
immutable checkpoint completed 10/10 after a stabilization wait. This is an
infrastructure retry, not a second model or rollout selection.

This establishes retention and action-dependent latent prediction at the halfway
checkpoint. It does not establish a control advantage because the action-only and
BF16-head comparisons also scored 10/10 under this small screen.
