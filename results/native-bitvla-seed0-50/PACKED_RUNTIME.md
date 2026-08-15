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

## Profile-guided direct-kernel follow-up

The exact runtime is the correctness baseline, not the final systems design. It
still reconstructs each packed matrix as BF16 immediately before `F.linear`. A
separate one-query PyTorch profiler run on an otherwise idle RTX PRO 6000
Blackwell pod confirms that this is the first computation to replace.

| Profiled item | Time (ms) | Calls |
| --- | ---: | ---: |
| CUDA self time, whole request | 96.803 | — |
| Weight quantization (CUDA total) | 45.236 | 366 |
| Activation quantization (CUDA total) | 13.675 | 366 |
| BF16 `mm` self CUDA time | 21.962 | 212 |
| `copy_` self CUDA time | 17.944 | 2,132 |
| CUDA launch CPU time | 37.413 | 10,268 |

The trace is an official `get_bitnet_vla_action` request with two zero-valued
224×224 RGB images, zero 8D proprioception, and the fixed LIBERO prompt used in
the benchmark. It is a 626-token text prefill: 512 expanded image tokens, 57
prompt/action/stop tokens, and no `M=1` generation loop. The text backbone has
30 layers at hidden size 2,560; its MLP matrices alone contain 1.593B of the
2.480B eligible packed weights (64.2%). The first direct kernel should therefore
cover the `M=626, (N,K)=(6912,2560)` gate/up matrices and `(2560,6912)` down
matrix, followed by `(2560,2560)` attention and the vision matrices.

The remote raw artifacts are deliberately outside Git:

- Chrome trace: `/data/users/ali/bitvla_benchmarks/profiles/packed-kernel-rtx6000-1-20260815/dense-native-626-prefill.trace.json`
- Profiler table: `/data/users/ali/bitvla_benchmarks/profiles/packed-kernel-rtx6000-1-20260815/dense-native-626-prefill.profiler-table.txt`

The compact reproducibility record is
[`packed_kernel_profile_rtx6000.json`](packed_kernel_profile_rtx6000.json). It
records the checkpoint, hardware, exact shape contract, profile metrics, and
mutation guard. Run its protocol only from an idle GPU pod, direct all compiler
caches to `/tmp`, and monkeypatch BitVLA's `update_auto_map` and
`check_model_logic_mismatch` helpers to no-ops before model initialization. Those
helpers otherwise modify the shared checkpoint.

The same protocol is packaged as
[`profile_bitvla_packed_baseline.py`](../../scripts/profile_bitvla_packed_baseline.py).
After syncing this repository revision to a GPU pod, execute it from BitVLA's
Python 3.10 environment with:

```bash
/data/users/ali/BitVLA/.venv/bin/python \
  /data/users/ali/BitWAM/scripts/profile_bitvla_packed_baseline.py \
  --upstream-root /data/users/ali/BitVLA \
  --checkpoint /data/users/ali/bitvla_runs/bitwam-world-posttrain--102000_chkpt \
  --output-dir /data/users/ali/bitvla_benchmarks/profiles/packed-kernel-rtx6000-1-20260815
```

Microsoft's official GPU BitNet implementation is a useful layout reference: it
packs 16 two-bit weights into a 32-bit word and uses `dp4a`, but its dispatcher
only supports hard-coded `M=1` GEMV shapes. BitVLA needs the prefill shapes above,
so its kernel cannot be used as a drop-in. The exactness gate for a new kernel is
the upstream operation order: FP32 per-token absmax scale, round-and-clamp to
INT8, BF16 dequantized activation and BF16 ternary-weight scale, BF16 Tensor Core
accumulation/output, then BF16 bias. First prove bit-exact actions against this
reference; only then record latency and run the ordered LIBERO-10 smoke gate.

That experiment batch is now recorded in
[`../packed-kernel/README.md`](../packed-kernel/README.md). The direct Triton
W2A8 and tiled-BF16 candidates both failed the exact-action gate and were
stopped before policy timing/LIBERO. The independent direct-load artifact did
succeed: on a fresh A100 process it reduced deployed bytes by 65.30%, CUDA peak
by 63.45%, and peak RSS by 46.60%, while load time was 2.37% slower and is not
claimed as a startup-speed improvement.
