# Packed one-bit BitWAM runtime

BitWAM now has a real packed inference path for BitVLA's ternary `BitLinear`
weights. The model remains a one-bit ternary network in the BitNet sense, while
each `{-1, 0, +1}` weight is represented by a two-bit code at runtime. Four codes
are stored per byte, then decoded by a generated CUDA kernel immediately before
the BF16 projection.

The recommended deployment point is **exact text packing**. It packs the 210 text
backbone matrices and leaves the 156 vision matrices in their original form. Both
100-query runs produced exactly the same 8-by-7 action tensor as the dense model,
and the official ordered LIBERO-10 smoke evaluation retained 10/10 successes.

| Runtime | Mean p50 (ms) | p50 range (ms) | Mean p95 (ms) | Loaded CUDA (GiB) | Query peak (GiB) | Action error | LIBERO-10 smoke |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense BitWAM | 108.69 | 108.61–108.76 | 114.93 | 5.433 | 6.032 | reference | 10/10 |
| Text packed, exact | **107.82** | 107.20–108.45 | 118.84 | **2.060** | **2.651** | **0** | **10/10** |
| Fully packed, exact | 114.36 | 112.01–116.70 | 128.46 | **1.427** | **2.019** | **0** | **10/10** |
| Fully packed, fused activation | 106.93 | 105.98–107.88 | 113.60 | 1.427 | 2.019 | max 0.084 | 9/10 |

Text packing reduces resident CUDA allocation by 62.08% and query peak by 56.04%.
Its mean p50 is 0.79% lower and its mean request latency is 0.58% lower than the
dense reference, although mean p95 is 3.40% higher. The latency result is therefore
a small measured median improvement, not a broad tail-latency claim.

Full exact packing reduces resident CUDA allocation by 73.73% and query peak by
66.52%, but adds 5.22% to median latency. It is useful when memory capacity matters
more than latency. Across all 366 layers it replaces 4,959,780,864 BF16 weight bytes
with 619,972,608 packed bytes, an exact 87.5% reduction for eligible matrices.

The fused-activation variant is an ablation, not a deployment recommendation. It
was 1.62% faster at the median after two robust runs, but changed the action tensor
and reproducibly missed the bottom-drawer task, finishing 9/10. An earlier attempt
also exposed a shape-cache limit; using a dynamic compiled activation fixed that
infrastructure error but did not restore the task. The native INT8 and compiled
projection alternatives were also slower than the selected text-packed path.

## Interpretation and limits

This is a systems result on one NVIDIA B200 and a quality-retention smoke test, not
a new statistical control comparison. The primary 50-rollout result remains 45/50
for ternary BitWAM. The 10/10 runs here use one ordered rollout per LIBERO-10 task
to reject runtime regressions.

The current implementation packs after the standard Hugging Face loader has read
the BF16 checkpoint. Steady-state and query memory are genuinely reduced, but the
5.403-GiB deployment artifact is unchanged and initialization still reaches about
6.07 GiB during dense-reference validation and packing. A serialized packed
checkpoint plus direct low-memory loader is still required to reduce disk size and
startup peak.

The implementation borrows the representation already present but inactive in
[BitVLA](https://github.com/ustcwhy/BitVLA), without copying its training recipe.
The current [Microsoft BitNet GPU kernels](https://github.com/microsoft/BitNet/tree/main/gpu)
target fixed-shape W2A8 GEMV, especially `M=1`; the BitVLA policy query includes a
full image/instruction sequence, so those kernels are not directly substituted.
This makes packed full-sequence VLA execution and the measured memory/latency/quality
frontier the relevant BitWAM systems contribution.

Reproduction configs are:

- `configs/bitvla-world-ternary-packed-text-benchmark.yaml`
- `configs/bitvla-world-ternary-packed-text-eval-10.yaml`
- `configs/bitvla-world-ternary-packed-exact-benchmark.yaml`
- `configs/bitvla-world-ternary-packed-exact-eval-10.yaml`
- `configs/bitvla-world-ternary-packed-fused-benchmark.yaml`
- `configs/bitvla-world-ternary-packed-fused-eval-10.yaml`

On the current cluster image, TorchInductor also needs the user-local Python 3.10
headers on `CPATH`:

```bash
export CPATH=/data/users/ali/.local/python-dev-3.10/usr/include:/data/users/ali/.local/python-dev-3.10/usr/include/python3.10
```

Compact aggregate data is in `packed_runtime_metrics.json`; per-query samples and
evaluation logs are retained under the ignored `raw/packed-runtime/` directory.
