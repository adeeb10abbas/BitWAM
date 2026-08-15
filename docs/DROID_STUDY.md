# DROID staged-training and world-action comparison

## Study question

Does large-scale DROID action-conditioned future prediction improve a compact
ternary BitVLA controller after matched LIBERO adaptation, and what memory and
latency trade-off does it make relative to released small world-action models?

The study transfers the training idea, not another model's implementation. The
BitWAM world objective remains a future latent target on the native BitVLA
backbone. DROID is standardized by BitVLA's pinned OXE transform to 7-D
base-frame delta translation, delta Euler rotation, and gripper action. DROID's
7-D proprio vector is not padded into LIBERO's 8-D state space: DROID stages omit
proprio, and the exact released LIBERO projector is restored for post-training.

## Data contract

| Item | Contract |
| --- | --- |
| Source | `gs://gresearch/robotics/droid/1.0.1/` |
| Immutable smoke slice | `gs://gresearch/robotics/droid_100/1.0.0/` |
| Views | one randomly selected exterior view plus wrist view |
| Action | 8 steps × 7 normalized base-frame controls |
| Target | primary-camera vision latent after all eight controls |
| Language | DROID `language_instruction` through the pinned OXE transform |
| Integrity | object size and MD5 verified; download manifest stored beside data |
| Training state | no DROID proprio; released LIBERO proprio restored only at Stage F |

The full release has no official validation split. Stage P and M use the
deterministic TFDS slice `train[:99%]`; `train[99%:]` is reserved before
normalization statistics, decoding, or augmentation. The smoke set is an
infrastructure check and must not be presented as an independent validation
result.

## Stages and gates

| Stage | Updates | Global batch | Trainable components | Gate |
| --- | ---: | ---: | --- | --- |
| S: DROID-100 smoke | 2 | 2 | BF16 world head | verified schema, finite loss, checkpoint |
| P: DROID pretrain | 20,000 | 256 | BF16 world head | held-out cosine improves over initialization and shuffled actions |
| M: DROID midtrain | 5,000 | 128 | action pathway + BF16 world head | positive action-conditioning gap without action regression collapse |
| F: LIBERO posttrain | 2,000 | 128 | action pathway + ternary world head | ordered 10-task smoke, then paired 50-episode evaluation |

Stage P runs on four B200s. Stage M and F use the same four-rank topology so
throughput comparisons are not confounded by hardware. Three seeds are promoted
only after seed 0 passes its stage gate.

## Required ablations

1. Released BitVLA, existing LIBERO-only action-only, BF16-world, and
   ternary-world runs.
2. DROID action-only mid/post-training with the same optimizer steps and images.
3. DROID world pretraining without Stage M, isolating the value of joint
   intermediate adaptation.
4. Zero-action and shuffled-action world controls, evaluated on the identical
   held-out episodes.
5. BF16 versus ternary world head with identical transferred backbone.
6. Cosmos3 Edge-Policy-DROID (4B) and Nano-Policy-DROID (16B) on a shared DROID
   episode manifest. Edge is the primary external size comparison.
7. Existing DROID baselines available on shared storage (pi0.5, GR00T N1.7, and
   DreamZero) are reported only where their input/output and evaluation contract
   matches; otherwise they remain contextual rows.

## Metrics and reporting rules

- Closed loop: LIBERO success with paired seeds and task order.
- World model: future-latent cosine/MSE and correct-vs-shuffled action gap on the
  DROID holdout.
- DROID action prediction: normalized and unnormalized L1 where dataset
  statistics are available.
- Systems: checkpoint bytes, theoretical packed bytes, peak allocated/reserved
  CUDA memory, steady resident memory, warmup-excluded latency, throughput, and
  energy/thermal telemetry when available.
- External models: report their native runtime and precision. Do not label a run
  TensorRT unless an actual, validated TensorRT engine produced the measurement.
- NVIDIA's current TensorRT-LLM Cosmos3 recipe covers audiovisual image/video
  generation, not the DROID action mode. Action baselines therefore use Cosmos
  Framework or vLLM-Omni unless action support is validated in the exact
  TensorRT-LLM revision under test.
- Comparisons between latent prediction and video generation are separated;
  video quality is not treated as the same target as BitWAM's latent objective.

Every result row must name the exact code revision, model/checkpoint hash,
dataset manifest, stage config, seed, GPU type/count, precision, and runtime.
