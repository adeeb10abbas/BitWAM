# Native BitVLA DROID-100 smoke

The DROID data path, future-frame target, frozen-controller world-head gradient,
and checkpoint writer passed on one NVIDIA B200. TFDS opened all 100 episodes
from the 2.04 GiB public sample after all 33 objects passed size and MD5 checks.

The valid run performed two optimizer steps from the inclusive upstream resume
boundary and wrote the action head, BF16 world head, optimizer, scheduler,
dataset statistics, processor, and provenance manifest at step 100001. Final
world loss was finite at 0.97842 and future-latent cosine was 0.02158, which is
an initialization smoke value rather than a quality result. The near-zero
correct-versus-shuffled gap is expected before DROID pretraining and must not be
reported as action conditioning.

The first launch is excluded as setup-invalid: running the upstream trainer
without `torchrun` left its unconditional distributed barrier uninitialized.
It exited before model training. The corrected one-rank `torchrun` launch
completed normally.

No policy-quality, compression, latency, or DROID generalization claim follows
from this smoke. Those claims require the preregistered full-data stages and
held-out/closed-loop evaluations in `docs/DROID_STUDY.md`.
