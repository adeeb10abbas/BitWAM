# BF16 baseline

## What we ran

We evaluated the published VLA-JEPA LIBERO checkpoint without compression. This was
the reference policy used to set the BitWAM success gate.

## Policy evaluation

- **49 successes from 50 episodes** across LIBERO-10.
- Nine tasks scored 5/5.
- Task 8 scored 4/5.
- All 50 rollout videos were written successfully.
- Evaluation finished normally in about six minutes on one RTX 3090.

## What this proves

The simulator, observations, checkpoint, and evaluation path can produce a highly
successful policy on this machine. A compressed BitWAM candidate must therefore be
compared against a working reference, not against a broken setup.

## Important limitation

This evaluation loaded the upstream VLA-JEPA policy directly. BitWAM's BF16 wrapper
matches it in parity tests, but a saved BitWAM BF16 checkpoint still needs its own
closed-loop rollout evaluation.

## Decision

Use 49/50 as the measured baseline. The short-pilot promotion gate is 45/50. Do not
claim that the saved BitWAM wrapper itself has completed closed-loop evaluation until
the cluster BF16 control is run.
