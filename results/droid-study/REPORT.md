# BitWAM DROID staged study

Status: running. Quality cells marked `pending` are intentionally not inferred
from the DROID-100 infrastructure smoke or from a different simulator arena.

## Question and primary comparison

The primary question is whether a compact ternary BitVLA controller benefits
from action-conditioned future-latent pretraining on DROID, followed by joint
DROID mid-training and matched LIBERO-10 post-training. The comparison holds
data, optimizer updates, global batch, and final task adaptation fixed wherever
the model interfaces permit it.

| Arm | DROID P | DROID M | LIBERO F | World action | Status |
| --- | --- | --- | --- | --- | --- |
| BitWAM staged | 20k frozen-head | 5k joint | 2k joint/ternary | observed | pending |
| BitWAM no-M | 20k frozen-head | none | 2k joint/ternary | observed | pending |
| Visual-only control | 20k frozen-head | none | matched downstream | zero | pending |
| Shuffled-action control | 20k frozen-head | none | none | within-rank permutation | pending |
| Action-only control | none | 5k action | 2k action | loss weight zero | pending |
| Released BitVLA | none | none | none | none | existing reference |

Seed 0 is a promotion gate. Three training seeds and the larger rollout tier are
run only after the seed-0 world, action, and closed-loop gates pass. The exact
stage definitions are preregistered in `docs/DROID_STUDY.md`.

## Data and evidence gates

| Gate | Evidence | Result |
| --- | --- | --- |
| DROID-100 object integrity | 33 objects; 2,192,615,094 bytes; per-object MD5 | passed |
| DROID-100 schema/gradient | two optimizer steps on one B200 | passed |
| Full DROID 1.0.1 object integrity | 2,050 objects; per-object size and MD5 | running |
| Full train statistics | deterministic `train[:99%]`, cache keyed by split | pending full transfer |
| Full-release gradient | two optimizer steps on one B200 | pending statistics |

The DROID-100 run produced finite losses and a checkpoint, but it is an
initialization smoke and not a quality result. Its archived action L1 was
0.33301, world loss 0.97842, future cosine 0.02158, and observed-minus-shuffled
conditioning gap approximately -0.00006.

## Metrics that will decide the claim

| Scope | Primary metrics |
| --- | --- |
| DROID holdout | future cosine/loss; observed-vs-shuffled action gap; normalized action L1 |
| LIBERO closed loop | paired task success with identical ordered initial states |
| Training systems | global examples/s; allocator peak allocated/reserved bytes; wall time |
| Deployment systems | warmup-excluded p50/p95; loaded/query-peak VRAM; deploy artifact bytes |
| Compression | exact-action packed storage and memory; accuracy-changing kernels separate |

Every DROID training metrics row records wall time, rank/world size, global
examples seen and per-second throughput, and CUDA current/peak allocation and
reservation. Raw A100 and B200 times are never compared as a model speed claim.

## External world-action context

The sealed rows below are retained as contextual evidence while the new common
DROID holdout is pending. They use a DROID/RoboLab closed-loop arena, not the
official-RLDS holdout, so their success counts are not pooled with BitWAM.

| Model | Class/interface | Valid success | Directional detail | Evidence SHA-256 |
| --- | --- | ---: | --- | --- |
| [Cosmos3 Edge Policy DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID) | WAM; 32 actions + 33 decoded frames | 6/6 | left 3/3, right 3/3 | `1c559ee5667ac9d22d7b66eafa7a65551783eedaf7fb3de29a2faf450c2dd029` |
| [Cosmos3 Nano Policy DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) | WAM; 32 actions + 33 decoded frames | 6/6 | left 3/3, right 3/3 | `4a6cc1d61593c7ba5272e1707f6bbe51261f7d23438070992bd75fd9e95fdb93` |
| Cosmos3 Nano, guidance 1 | WAM guidance ablation | 4/6 | left 1/3, right 3/3 | `8796f4ab9ea9490ee5b78678bc689d6fd13f27a9551006f8b5d346e202d0cc5c` |
| DreamZero DROID | joint action/latent-video | 3/6 | left 2/3, right 1/3 | `4c76cdc3ca9eaf227d21d160199408f22e1b3dd7a71176a5a5dbe22223714461` |
| GR00T N1.7 DROID | VLA | 0/6 | pickup/interaction 3/3 each direction | `95077a42bb0115bc673ea13ae5acdc6fdef6f476627804662f73c219ebd88bc7` |
| pi0-FAST DROID | VLA | 11/20 | left 1/10, right 10/10 | `491c74812ed0e4d36c16f8e0ded17a70af3e69740c9bcb87af129bb6d9563073` |

The locally measured Cosmos3 Edge fixed-observation request was 5.526 seconds
for 32 actions plus 33 frames, with a 14,435 MiB steady-state policy-GPU point
measurement. That value is not a peak. BitWAM predicts a future latent during
training and discards the auxiliary head for control deployment, whereas Cosmos
emits decoded video, so the two latency scopes remain separate.

Cosmos policy checkpoints also use their native action representation and
runtime. [NVIDIA's DROID recipe](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md)
predicts eight-dimensional absolute joint actions; BitWAM uses BitVLA's
seven-dimensional base-frame delta transform. Action L1 is therefore reported
within each representation, never directly across them.

The current official [Cosmos inference documentation](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/inference.md)
and [TensorRT-LLM cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/README.md)
cover TensorRT audiovisual generation, not the DROID action-policy path. Cosmos
action measurements are labeled as Cosmos Framework or vLLM-Omni unless an
action-capable TensorRT engine is actually validated. No framework label is
inferred from checkpoint format alone.

### Smaller released WAM scope

"Small" is not normalized to the lowest number quoted by a paper. Trainable,
active, frozen-backbone, and total deployed parameters are retained as different
quantities. The common DROID comparison is limited to models with a compatible
released DROID policy; the other rows remain arena-separated context.

| Model | Release scale/interface | Runnable evidence in this study | Comparison status |
| --- | --- | --- | --- |
| BitWAM | native ternary BitVLA action path plus 21.05 MiB training-only latent head | DROID/LIBERO staged study | primary |
| [UVA](https://github.com/ShuangLI59/unified_video_action) | compact joint video/action model; two-stage video then joint training | released LIBERO checkpoint, no common DROID arm | contextual |
| [Light-WAM](https://arxiv.org/abs/2606.08242) | 0.44B trainable plus frozen 1.3B Wan backbone | release reviewed; local RoboTwin gate only | next lightweight replication |
| [Efficient-WAM](https://efficientwam.github.io/) | 1B model; low-cost coarse future branch | sealed Efficient-WAM-RT RoboTwin trajectories | contextual, different arena |
| [Fast-WAM](https://github.com/yuantianyuan01/FastWAM) | action-only test-time mode; total deployed scale not normalized here | sealed RoboTwin trajectories | contextual, different arena |
| Cosmos3 Edge Policy DROID | 4B policy model card; decoded video/action output | sealed DROID/RoboLab plus pending common manifest | primary external DROID arm |
| Cosmos3 Nano Policy DROID | 16B policy model card; decoded video/action output | sealed DROID/RoboLab plus pending common manifest | larger reference, not a small arm |

Efficient-WAM's official project reports a 1B model and roughly 100 ms physical
deployment chunks, while the released Fast-WAM recipe trains on eight GPUs for
LIBERO and defaults evaluation to eight GPUs. Those source-native figures are
not substituted for measurements on BitWAM's B200 protocol.

## Existing deployment reference

Before DROID training, exact text packing of the ternary BitVLA action path
preserved actions and the ordered 10/10 smoke result while reducing resident CUDA
allocation from 5.433 GiB to 2.060 GiB (62.08%) and query peak from 6.032 GiB to
2.651 GiB (56.04%). Mean p50 changed from 108.69 ms to 107.82 ms; p95 increased
from 114.93 ms to 118.84 ms. This is a demonstrated memory improvement and a
small median measurement, not a tail-latency speedup. The DROID-final checkpoint
will be rerun under the identical two-process protocol.

## Claim boundary

No DROID quality improvement is claimed until the full-release holdout and
matched downstream controls finish. A positive result requires improvement over
the action-only arm, not merely over the released checkpoint. A negative result
is retained: a compact model that compresses well but fails action-conditioning
or closed-loop gates is reported as such rather than selected post hoc.
