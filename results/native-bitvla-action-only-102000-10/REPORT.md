# Action-only native ternary control at 2,000 updates

The full compute-matched action-only control completed all 10 LIBERO-10 smoke
rollouts, one trial per task. It used the same examples, action horizon, future-image
encoding path, effective global batch, policy learning rate, and 2,000 update budget
as the primary BitWAM run, but its world-loss coefficient was zero.

The unused random predictor retained a near-zero correct-vs-shuffled action gap.
This run shows that continued native ternary training preserves closed-loop behavior;
it does not establish any world-model benefit. The primary comparison is the saved
2,000-update ternary BitWAM checkpoint under the identical rollout ordering.
