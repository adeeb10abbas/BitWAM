# Action-only native ternary control at 1,000 updates

The compute-matched action-only control completed all 10 LIBERO-10 smoke rollouts,
one trial per task. It used the same data path, future-image encoding, batch, learning
rates, and checkpoint format as BitWAM, but the world-loss coefficient was zero.

The unused random predictor had a near-zero correct-vs-shuffled action gap, as
expected. This result establishes that conservative continued training of the native
ternary policy preserves closed-loop behavior. It does not establish a world-model
benefit; the ternary BitWAM checkpoint must be compared against this control.
