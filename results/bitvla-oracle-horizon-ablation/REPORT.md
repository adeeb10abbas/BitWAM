# Receding-horizon oracle ablation

## Verdict

Yes: extending the score horizon from 8 to 32 actions is worth pursuing. It
produced two candidate-associated rescues and no candidate-associated harms on the
four difficult LIBERO-10 tasks. However, extending to 64 actions was worse. The
useful design is therefore **plan 32, execute 8, then replan**—not an arbitrarily
long action chunk.

This is directional, exploratory evidence rather than a paper-ready success-rate
claim. H=32 improved only one net episode out of 20, with a paired interval that
still includes zero.

| Score horizon | Success | Delta vs H=8 | Strict selections | Branch work |
| ---: | ---: | ---: | ---: | ---: |
| 8 actions | 17/20 (85%) | — | 15 | 1.00x |
| 32 actions | 18/20 (90%) | +5 points | 23 | 3.74x |
| 64 actions | 15/20 (75%) | -10 points | 24 | 7.72x |

The H=32 minus H=8 paired bootstrap 95% interval is `[-10, +20]` percentage
points and exact McNemar `p=1.0`. H=64 minus H=32 is -15 points with interval
`[-30, 0]` and McNemar `p=0.25`. The selected hard-task sample is too small for
statistical resolution.

## What changed

Candidate zero begins with the policy's exact eight-action chunk. For horizons 32
and 64, an isolated rendered simulator follows that chunk and asks the same policy
for another action chunk every eight steps, producing one shared nominal future.
Sixteen nested perturbations of that complete plan are then scored in exact,
camera-free MuJoCo physics. Only the first eight actions of the selected plan are
executed in the control environment before replanning.

This preserves short feedback intervals while allowing the selector to see delayed
consequences. It does not execute 32 or 64 actions open loop.

The tasks were `T0`, `T6`, `T7`, and `T8`, five initial states each. They were
chosen adaptively because they contained the informative failures in the first
best-of-K screen. H=8 and H=32 were rerun or retained on the same physical A100 for
each task. H=64 ran concurrently on separate A100s as a negative-control horizon.

## Mechanistic result

H=32 changed three episode endpoints relative to H=8:

- `T6/e4`: H=8 timed out at 530 steps after one strict local selection. H=32
  made 12 strict long-horizon selections and succeeded in 281 steps.
- `T8/e2`: H=8 timed out with zero strict selections. H=32 made one strict
  selection and succeeded in 375 steps.
- `T8/e4`: H=8 succeeded and H=32 failed, but neither trajectory made any
  candidate substitution. This is a no-selection rollout flip, not evidence that
  the 32-action selector chose a harmful candidate.

The `T6/e4` H=8 failure was repeated on the exact H=32 GPU and reproduced. The
`T8/e2` comparison was already physically GPU-paired. Both candidate-associated
rescues therefore survive the hardware-pairing check.

H=64 made one rescue but three candidate-associated harms relative to H=8. It
also failed `T8/e2`, which H=32 rescued. A shared nominal continuation becomes a
poor guide that far into the future: perturbations alter the early state, while
their later actions remain based on the unperturbed nominal trajectory.

## Decision

Use a 32-action target for the next value-model experiment and continue executing
only eight actions between observations. Do not use H=64 with this shared-tail
candidate construction.

The next minimal reliability gate is more initial states on `T6` and `T8`, paired
H=8 versus H=32 on the same GPUs. If the rescue rate persists, train a small
terminal-success or remaining-steps head at the 32-action horizon and replace the
exact simulator score with that learned value. Only after that should BitWAM be
benchmarked under the fixed 8 GiB, power, and control-deadline budget.

The branch-work and wall-time ratios here belong to the exact MuJoCo diagnostic.
They are not BitWAM inference-speed measurements.
