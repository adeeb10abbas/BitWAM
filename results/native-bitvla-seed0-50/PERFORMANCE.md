# Native BitVLA and BitWAM systems profile

> Follow-up: [`PACKED_RUNTIME.md`](PACKED_RUNTIME.md) reports the implemented packed
> inference runtime. Its recommended exact text-packed mode reduces resident CUDA
> memory by 62.08%, preserves 10/10 smoke success, and measures a 0.79% lower p50.
> The measurements below describe the earlier unpacked checkpoints.

The saved seed-0 policies were measured in two independent processes each on the
same NVIDIA B200. Every run used batch size 1, a fixed seed-0 pair of 224x224 RGB
images, zero proprioception, 20 warmup queries, and 100 timed queries. Timing covers
the official `get_action` request, including input preprocessing and synchronized
CUDA inference. Each request produces an eight-action chunk.

| Deployed policy | Mean p50 (ms) | p50 range (ms) | Mean p95 (ms) | Loaded VRAM (GiB) | Peak VRAM (GiB) | Deploy files (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Released BitVLA | 109.58 | 109.26–109.90 | 115.52 | 5.433 | 6.032 | 5.403 |
| Action-only | 110.54 | 109.61–111.48 | 116.65 | 5.433 | 6.032 | 5.403 |
| BitWAM-Ternary | 108.69 | 108.61–108.76 | 114.93 | 5.433 | 6.032 | 5.403 |
| BitWAM-BF16 | 109.30 | 109.04–109.56 | 117.90 | 5.433 | 6.032 | 5.403 |

The measured ternary p50 is 0.82% below the released controller, but this is not a
BitWAM speedup claim. The predictor is absent from deployment, and all four runs
have exactly the same action-path parameter count, tensor storage, loaded CUDA
allocation, query peak, and dense execution graph. The observed latency spread is
small process-level variation among differently trained weights on the same graph.
The defensible result is zero added deployment cost, not accelerated deployment.

The evaluator-loaded artifact set is 5,801,049,511 bytes for each post-trained
policy (the released set differs by only 570 serialization bytes). The 22,072,975-byte
world predictor and 11,611,918,629-byte optimizer state are training-only and can be
omitted from a deployment package. Omitting them is packaging hygiene rather than a
new compression result.

The completed 2,000-update logs also show nearly identical end-to-end job time:
73.26 minutes for action-only, 73.30 minutes for ternary BitWAM, and 72.90 minutes
for BF16 BitWAM. This scope runs from `torchrun` startup through final checkpoint
save on two B200s. The two world-head jobs overlapped on separate GPU pairs in the
same pod, while action-only ran earlier, so the sub-percent ordering is not a
throughput claim. It does show no material wall-clock penalty at this scale. A
full-training peak allocator trace was not archived and is therefore not reported.

## Auxiliary predictor

The training-only predictor was also isolated at the real dimensions with batch 8.
Each timed step includes the correct-action and shuffled-action world forwards plus
backward; the optimizer is excluded.

| Predictor | Stored parameters (MiB) | p50 range (ms) | p95 range (ms) | Peak allocated (MiB) | Temporary increment (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 21.046 | 3.289–3.300 | 4.007–6.370 | 81.825 | 22.129 |
| Ternary QAT simulation | 21.046 | 3.357–9.498 | 8.157–10.888 | 131.344 | 71.649 |

The current ternary implementation keeps BF16 master weights and performs absmean
quantization on the fly. It therefore realizes neither a storage reduction nor a
speedup; its temporary allocation is 3.24 times the BF16 increment, and its timing
is slower and less stable. A packed representation could reduce total predictor
parameter storage from 21.046 MiB to about 2.647 MiB with simple two-bit matrices
(87.42%) or 2.102 MiB with ideal ternary entropy coding (90.01%), including the
unpacked norm and bias terms. Those are theoretical storage bounds until a packed
checkpoint and compatible kernel exist.

Compact measurements are in `performance_metrics.json`; per-query samples and job
logs remain in the ignored `performance/raw/` directory.
