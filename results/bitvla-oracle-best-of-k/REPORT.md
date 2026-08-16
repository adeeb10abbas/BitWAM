# Best-of-K one-chunk oracle: fast go/no-go result

## Verdict

The proposed shortcut does **not** yet justify training a Q function. K=16 reached
47/50 versus 46/50 for K=1, but the paired improvement is only +2 percentage
points with a 95% bootstrap interval of `[-4, +10]` and exact McNemar `p=1.0`.
K=4 was slightly worse at 45/50.

The useful negative result is more specific: exact physics regularly found an
action chunk with better immediate LIBERO predicate progress, yet greedily taking
that chunk did not reliably improve the completed task. Ternary memory savings can
make more candidates affordable, but candidate count is not the missing ingredient.
The missing ingredient is a longer-horizon value signal.

## Minimal experiment

For each of five ordered initial states on every LIBERO-10 task, candidate zero was
the policy's exact eight-action chunk. The remaining candidates were nested,
temporally smooth perturbations of the six continuous controls at 0.15 times the
training action standard deviation. Gripper commands were unchanged and actions
were clipped to training quantiles.

At each policy decision, an isolated camera-free MuJoCo process restored the exact
main-environment state, executed every candidate, and counted the official LIBERO
goal predicates after the chunk. A candidate replaced the policy output only when
its exact score was strictly higher; ties retained the base action. This is a
diagnostic for a perfect *local* selector, not a test of the learned BitWAM predictor
and not an upper bound on long-horizon planning.

All ten task shards ran in parallel on ten A100s. Each task stayed on the same GPU
for K=1, K=4, and K=16.

Each shard used this command shape, with `TASK`, `K`, and `OUTPUT` replaced by the
manifested values:

```bash
python -m lerobot_policy_bitwam.bitvla_oracle \
  --config configs/bitvla-oracle-libero10.yaml \
  --task-id TASK --candidate-count K --trials-per-task 5 \
  --output-path OUTPUT
```

| Candidates | Success | Delta vs K=1 | Strict local selections | Max shard time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 46/50 (92%) | — | 0 | 131 s (1.00x) |
| 4 | 45/50 (90%) | -2 points | 28 | 290 s (2.21x) |
| 16 | 47/50 (94%) | +2 points | 28 | 603 s (4.59x) |

The wall-time column measures this exact MuJoCo oracle harness, not a packed-WAM
latency claim. It is included only as an experiment sanity check.

Per-task successes:

| K | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 2 | 5 |
| 4 | 4 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 2 | 5 |
| 16 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 5 | 3 | 5 |

## Why the one-episode gain is not persuasive

K=16 versus K=1 had two apparent rescues and one harm. One rescue, task 8 episode
4, made no candidate substitution at all, so it is evaluator/model nondeterminism.
Among endpoint flips where a strict candidate was actually selected, task 0 episode
2 was rescued and task 6 episode 4 was harmed: net zero. K=4 similarly harmed task
7 episode 1 after three locally favorable substitutions.

The same checkpoint previously scored 45/50 in the standard run recorded in
`results/native-bitvla-seed0-50/metrics.json`; the no-branch K=1 repeat here scored
46/50. That 45–46 aggregate repeat band and the no-substitution task-8 flips are why
the raw 47/50 must not be presented as a planning gain.

## Decision

Do not build a broad Dreamer-style Q-learning stack from this result. The smallest
next research gate, if pursued, is to score the same shared-prefix candidates with
a terminal-success or remaining-steps value target over a longer horizon. The paper
claim becomes interesting only if that value-guided selector produces a repeatable
closed-loop gain under the same memory, power, and deadline budget. More one-chunk
candidates with immediate predicate scoring are insufficient.
