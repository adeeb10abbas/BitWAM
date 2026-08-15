# Frozen-controller world pretraining at 1,000 updates

The future-latent predictor reached 0.9206 mean cosine similarity in the final 100
metric records before the checkpoint. The saved action head and proprio projector
remain bit-identical to the released native BitVLA control.

This establishes a learned future-prediction signal without changing closed-loop
policy behavior. It is not evidence that world-supervised joint post-training improves
task success; that claim requires the matched LIBERO evaluations.
