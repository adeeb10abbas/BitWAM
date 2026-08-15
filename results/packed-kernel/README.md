# Direct packed-kernel experiment batch

Status: the kernel and packed artifact are implemented and tested, but the
direct compute backend is **not approved for policy evaluation**. Every direct
candidate failed the exact-action gate and was stopped before whole-policy
timing or LIBERO evaluation.

## Profile and workload

One official dense BitVLA request on an RTX PRO 6000 Blackwell used a 626-token
text prefill. Across the request, 366 weight-quantization calls consumed 45.236
ms CUDA-total, 366 activation-quantization calls consumed 13.675 ms, BF16 `mm`
consumed 21.962 ms self CUDA, and 10,268 CUDA launches cost 37.413 ms of CPU
launch time. The text MLP owns 1.593B of 2.480B packable weights (64.2%). See
`../native-bitvla-seed0-50/packed_kernel_profile_rtx6000.json`.

## Direct W2A8 results

All values below include activation quantization plus one projection and use
CUDA-event p50 latency. `speedup` is dense BF16 latency divided by direct packed
latency. The tuned scalar kernel reads flat two-bit weights directly, decodes
only register tiles, and never allocates a complete decoded weight matrix.

| GPU / path | M,K,N | Dense (ms) | Direct (ms) | Speedup | BF16 gate |
| --- | --- | ---: | ---: | ---: | --- |
| RTX 6000, exact PyTorch activation, scalar 64×256×64 | 626,2560,2560 | 0.1509 | 0.2024 | 0.746× | fail |
| RTX 6000, exact PyTorch activation, scalar 64×256×64 | 626,2560,6912 | 0.1411 | 0.3264 | 0.432× | fail |
| RTX 6000, exact PyTorch activation, scalar 64×256×64 | 626,6912,2560 | 0.3317 | 0.3367 | 0.985× | fail |
| RTX 6000, exact hybrid activation, scalar 64×256×64 | 626,2560,2560 | 0.1488 | 0.2086 | 0.714× | fail |
| RTX 6000, exact hybrid activation, scalar 64×256×64 | 626,2560,6912 | 0.1401 | 0.3337 | 0.420× | fail |
| RTX 6000, exact hybrid activation, scalar 64×256×64 | 626,6912,2560 | 0.3302 | 0.3342 | 0.988× | fail |
| A100 40GB, exact PyTorch activation, scalar 64×256×64 | 626,2560,2560 | 0.1802 | 0.4424 | 0.407× | fail |
| A100 40GB, exact PyTorch activation, scalar 64×256×64 | 626,2560,6912 | 0.3031 | 0.9211 | 0.329× | fail |
| A100 40GB, exact PyTorch activation, scalar 64×256×64 | 626,6912,2560 | 0.3533 | 0.6779 | 0.521× | fail |

The all-Triton activation prepass reached a 1.059× layer speedup for the RTX
down projection, but real-shape activation codes/scales were not bit-exact to
ATen and the result is retained only as a rejected ablation. The hybrid path
keeps ATen's FP32 scale reduction and produced zero activation code/scale
differences at K=37, 2560, and 6912 on RTX and B200. It still cannot reproduce
the upstream BF16 GEMM accumulation order. The byte-once `lane4` decoder was
also slower on RTX and B200.

## Exact-action gates

The dense repeat was bit-exact in every gate. Direct candidates were applied
only to the text tower or its 30 MLP down projections. All were rejected before
latency measurement:

| Candidate | Scope | Max action error | Mean action error |
| --- | --- | ---: | ---: |
| W2A8 INT32 accumulation | text | 0.053153 | 0.014535 |
| tiled BF16 candidate | text | 0.018271 | 0.005021 |
| W2A8 INT32 accumulation | text MLP down | 0.033203 | 0.008702 |
| tiled BF16 candidate | text MLP down | 0.064453 | 0.006917 |

The four `action-gate-*.json` files contain the machine-readable gate records.
No direct candidate was promoted to LIBERO because action equality is the first
quality gate.

## Direct packed artifact

The real 2.8B-parameter BitVLA topology can now be constructed on `meta` and
loaded directly from a versioned, hash-validated packed artifact without
reading the dense model safetensors.

| Fresh A100 load | Dense | Packed direct | Change |
| --- | ---: | ---: | ---: |
| deployed model files | 5,639,200,392 B | 1,956,765,207 B | −65.30% |
| CUDA peak | 5,737,242,112 B | 2,096,799,744 B | −63.45% |
| process peak RSS | 6,960,940 KiB | 3,717,352 KiB | −46.60% |
| load time | 23.934 s | 24.502 s | +2.37% |

The load leaves zero meta parameters and a real 1152→1152 packed BitLinear
forward produced finite output. Startup latency is not yet improved; replacing
the Python tensor archive and eager per-tensor hashing with a mmap-friendly
format is the next artifact optimization.

## Reproducibility

- `benchmark_packed_kernel.py` runs real-shape tile sweeps.
- `profile_bitvla_packed_baseline.py` reproduces the request profile without
  modifying the shared checkpoint.
- `validate_packed_artifact.py` measures dense and direct-load memory in fresh
  processes.
- `rtx6000-text-tile-sweep.json`, `rtx6000-text-lane4.json`,
  `rtx6000-text-hybrid.json`, and `a100-40gb-text-scalar.json` retain the main
  kernel measurements.
- `packed-artifact-{dense,direct}-a100.json` retain the fresh-load measurements.

The next credible compute path needs a CUTLASS/CUDA mainloop that consumes a
kernel-friendly pre-permuted packed layout and reproduces the same BF16 Tensor
Core operation order. The present Triton W2A8 implementation proves direct
packed execution and exposes its limits; it does not justify a speed or quality
claim for the paper.
