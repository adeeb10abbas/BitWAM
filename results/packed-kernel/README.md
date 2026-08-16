# Direct packed-kernel experiment batch

Status: a production W2A8 CUDA backend now consumes DP4A-interleaved two-bit
weights directly and has passed layer-level CUDA correctness, full-policy
timing, and a real kernel-matched recovery-training smoke test. It is **not yet
approved for a task-success claim**: the un-recovered checkpoint's actions are
not identical to the upstream BF16 operation order, and closed-loop LIBERO
evaluation must follow recovery training.

## Production BitNet-style CUDA path (2026-08-15)

The implementation adapts the packed interleave and integer dot-product
principles from Microsoft's MIT-licensed [BitNet GPU implementation][bitnet-gpu]
at commit `0b341e582afbf9e1011f24744b554c96a3477eb5`. It does not vendor that
shape-static kernel. BitWAM implements a shape-generic Triton integration with:

- a 16-weight, four-byte packed permutation performed directly packed-to-packed;
- a DP4A mainloop for one to eight tokens;
- a Tensor-Core INT8 mainloop for image/text prefill batches;
- INT32 accumulation and one FP32 scale/bias epilogue; and
- zero dense BF16 or INT8 weight-cache bytes.

This is the CUDA approach described at [bitnet.live][bitnet-live], adapted to
BitVLA's real projection shapes rather than its published static M=1 shapes.

### Paired end-to-end A100 result

Two independent paired processes used the same A100-SXM4-80GB, checkpoint,
seeded 224x224 two-camera observation, official `get_action`, ten warmups, and
50 synchronized timed queries per process. The dense control materializes each
ternary weight once to BF16 and reuses it, so it is the fair latency control;
the upstream dynamic path is slower because it re-quantizes weights per call.

| Runtime | p50 across process runs | Loaded CUDA allocation | Query peak | Action difference |
| --- | ---: | ---: | ---: | ---: |
| cached ternary BF16 | 136.52-137.22 ms | 5,880,098,816 B | 6,515,878,400 B | exact |
| packed W2A8 | 141.72-142.70 ms | 1,533,362,688 B | 2,168,180,736 B | max 0.12402, mean 0.01736 |

Across the two paired runs, packed W2A8 is 3.81-4.00% slower at p50 (3.90%
mean overhead). It reduces loaded CUDA allocation by 73.92% (3.83x capacity)
and query peak by 66.72% (3.01x capacity). Its 2,479,890,432 eligible weights
occupy 619,972,608 bytes instead of 4,959,780,864 BF16 bytes: an exact 8x,
87.5% reduction for those weights. Relative to the unfused upstream dynamic
runtime (230.84 ms p50), W2A8 is 1.62x faster, but that is not the fair
kernel-to-kernel baseline.

The A100's optimized BF16 GEMMs remain slightly faster because packed decode
and activation quantization add instructions; two-bit storage does not imply a
native two-bit Tensor Core instruction. The important current result is near
latency parity at roughly one quarter of steady device allocation—not a speed
breakthrough.

### Fresh DP4A artifact load

The production DP4A artifact was also validated in a fresh process, rather
than inferred from the in-memory conversion benchmark. Alias-aware export
preserves the tied token embedding / language-model head instead of storing it
twice. The artifact contains `1,300,064,754` bytes, loads with zero remaining
meta parameters, installs the W2A8 runtime on all 366 layers, and produces a
finite 1152-to-1152 projection directly through the packed CUDA kernel. Against
the fresh dense source load:

| Fresh A100 load | Dense | DP4A packed W2A8 | Change |
| --- | ---: | ---: | ---: |
| model weight artifact | 5,639,200,392 B | 1,300,064,754 B | -76.95% |
| CUDA peak | 5,737,242,112 B | 1,426,578,432 B | -75.13% |
| process peak RSS | 6,964,084 KiB | 3,079,436 KiB | -55.78% |
| load time | 26.449 s | 22.444 s | -15.14% |

Load time is sensitive to filesystem cache state; size, CUDA allocation, zero
meta tensors, layout metadata, and the direct-kernel forward are the acceptance
gates.

The same artifact then completed the full official policy query benchmark. It
loaded in 5.956 s, held 1,559,416,320 CUDA bytes after load, peaked at
2,205,140,992 bytes during `get_action`, and measured 144.18 ms p50 across ten
warmups and 50 synchronized queries. Relative to the cached-BF16 control, this
deployable path reduces loaded allocation 73.48% (3.77x capacity) and query peak
66.16% (2.95x capacity). The artifact export itself is 33.56% smaller than the
first DP4A export because tied tensors are no longer duplicated.

### Kernel-matched recovery smoke

`bitvla-world-w2a8-kernel-recovery-smoke.yaml` ran two real LIBERO micro-steps
through all 366 W2A8 BitLinear layers, backward propagation, optimizer updates,
and checkpoint save on an A100. The final recorded values were finite:

- action loss: `0.0449829`;
- world loss: `0.0487202`;
- future-latent cosine similarity: `0.951280`; and
- peak CUDA allocation: `38,325,069,312` bytes.

This proves that recovery can train against the deployed integer operation
order. It does not prove recovered closed-loop success. The full recovery is
configured to start from the DROID-to-LIBERO `107000` checkpoint after the live
four-B200 pipeline produces it.

### Claim boundary

The defensible systems claim today is: direct packed W2A8 compresses 366
text-and-vision BitLinear layers, cuts steady CUDA allocation 73.9%, and stays
within 4.0% of a cached-BF16 control. The action head, final outputs, and other
non-BitLinear modules are still BF16; the training world head uses ternary QAT
linears but is not part of deployed inference.

The memory result suggests that three isolated packed query peaks could fit in
the space of one 8 GiB device where only one dense peak fits. That is a capacity
hypothesis, not yet the paper's `X`: serially evaluating three futures would
spend roughly three times the compute. A multi-future batched/parallel benchmark
under fixed 8 GiB, power, and deadline is required before claiming `X`, and
closed-loop evaluation after recovery is required before claiming `Y`.

Machine-readable current results are in `w2a8-production-a100.json`.

## Earlier prototype evidence (superseded by the production path above)

The experiments below document why the original scalar packed candidates were
rejected. They remain useful negative results, but their conclusion that a new
CUTLASS/CUDA mainloop was still needed no longer describes the current code.

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

The production path above supplies the previously missing interleaved DP4A and
Tensor-Core mainloops. Exact BF16 action parity is intentionally no longer the
training target; kernel-matched QAT plus closed-loop success is the remaining
quality gate.

[bitnet-gpu]: https://github.com/microsoft/BitNet/tree/main/gpu
[bitnet-live]: https://bitnet.live/
