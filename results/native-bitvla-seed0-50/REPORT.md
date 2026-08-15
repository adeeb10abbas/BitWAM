# Native BitVLA seed-0 paired 50-rollout comparison

All four methods completed five ordered rollouts on each LIBERO-10 task using the
same evaluator seed and initial-state ordering.

| Method | Successes | Rate |
| --- | ---: | ---: |
| Released BitVLA | 39 / 50 | 78% |
| Action-only, 2,000 updates | 44 / 50 | 88% |
| BitWAM, ternary head | 45 / 50 | 90% |
| BitWAM, BF16 head | 47 / 50 | 94% |

Ternary BitWAM meets the predeclared 45/50 compressed-policy gate and improves by
one episode over the matched action-only run. That two-point difference is not
statistically resolved in this seed: the paired bootstrap 95% interval is
`[-8, +12]` percentage points and the exact McNemar p-value is 1.0. Ternary versus
released control is +12 points with paired bootstrap interval `[+2, +22]`, but its
exact McNemar p-value is 0.0703.

BF16 has the highest aggregate rate, but its +4 points over the ternary predictor
also remain unresolved (paired interval `[-4, +12]`, McNemar p=0.625). The evidence
therefore supports the claims that ternary BitWAM works, meets the retention gate,
and learns action-dependent latent prediction. It does not yet support a control
improvement over action-only or parity with the BF16 predictor across seeds.

Per-task successes out of five, in evaluator order:

| Method | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Released | 5 | 5 | 5 | 4 | 3 | 5 | 3 | 3 | 1 | 5 |
| Action-only | 5 | 5 | 5 | 4 | 4 | 5 | 5 | 4 | 3 | 4 |
| Ternary | 4 | 5 | 5 | 4 | 5 | 5 | 4 | 5 | 3 | 5 |
| BF16 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 2 | 5 |

This is a single training seed and must not be presented as the final paper matrix.

The matched B200 latency, VRAM, deploy-size, and auxiliary-head measurements are in
[`PERFORMANCE.md`](PERFORMANCE.md). A subsequent implementation and systems study
is in [`PACKED_RUNTIME.md`](PACKED_RUNTIME.md): exact text packing preserves 10/10
ordered smoke success while reducing resident CUDA memory by 62.08% and measuring
a small 0.79% p50 improvement. It does not change the single-seed statistical limits
of the 50-rollout control result above.
