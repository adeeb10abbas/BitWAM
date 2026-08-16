#!/usr/bin/env python3
"""Benchmark BitWAM's direct packed W2A8 projection on real BitVLA shapes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from lerobot_policy_bitwam.bitvla_packed_kernel import (
    PackedDp4aConfig,
    PackedKernelConfig,
    direct_packed_dp4a_int8_linear,
    direct_packed_int8_linear,
    direct_packed_w2a8_linear,
    repack_packed_ternary_for_dp4a,
)
from lerobot_policy_bitwam.bitvla_packing import quantize_activation

DEFAULT_SHAPES = (
    (626, 2560, 2560),
    (626, 2560, 6912),
    (626, 6912, 2560),
    (626, 2560, 640),
    (512, 1152, 1152),
    (512, 1152, 4304),
    (512, 4304, 1152),
)


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(part) for part in value.lower().replace("x", ",").split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be M,K,N") from error
    if len(shape) != 3 or min(shape) < 1:
        raise argparse.ArgumentTypeError("shape must contain three positive integers: M,K,N")
    return shape


def _parse_kernel_config(value: str) -> PackedKernelConfig:
    try:
        fields = tuple(int(part) for part in value.lower().replace("x", ",").split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("kernel config must be BLOCK_M,BLOCK_N,BLOCK_K,WARPS") from error
    if len(fields) != 4:
        raise argparse.ArgumentTypeError("kernel config must be BLOCK_M,BLOCK_N,BLOCK_K,WARPS")
    config = PackedKernelConfig(*fields)
    try:
        config.validate()
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return config


def _parse_dp4a_config(value: str) -> PackedDp4aConfig:
    try:
        fields = tuple(int(part) for part in value.lower().replace("x", ",").split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("DP4A config must be BLOCK_N,BLOCK_GROUPS,WARPS") from error
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("DP4A config must be BLOCK_N,BLOCK_GROUPS,WARPS")
    config = PackedDp4aConfig(*fields)
    try:
        config.validate()
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return config


def _pack_ternary(codes: torch.Tensor) -> torch.Tensor:
    flattened = codes.flatten().add(1).to(torch.uint8)
    padding = (-flattened.numel()) % 4
    if padding:
        flattened = F.pad(flattened, (0, padding))
    values = flattened.view(-1, 4)
    return values[:, 0] | values[:, 1] << 2 | values[:, 2] << 4 | values[:, 3] << 6


def _latency_ms(callable_: Any, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "min_ms": ordered[0],
    }


@torch.inference_mode()
def benchmark_shape(
    shape: tuple[int, int, int],
    *,
    warmup: int,
    iterations: int,
    seed: int,
    kernel_config: PackedKernelConfig | None,
    dp4a_config: PackedDp4aConfig | None,
    activation_backend: str,
    decode_mode: str,
    kernel_family: str,
) -> dict[str, Any]:
    m, k, n = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    inputs = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    codes = torch.randint(-1, 2, (n, k), device="cuda", dtype=torch.int8, generator=generator)
    weight_step = torch.tensor(0.0125, device="cuda", dtype=torch.float32)
    row_major_packed = _pack_ternary(codes)
    packed = (
        repack_packed_ternary_for_dp4a(row_major_packed, n, k)
        if kernel_family in {"auto", "dp4a"}
        else row_major_packed
    )
    dense_weight = (codes.float() * weight_step).to(torch.bfloat16)

    def dense() -> torch.Tensor:
        return F.linear(quantize_activation(inputs), dense_weight)

    def direct() -> torch.Tensor:
        if kernel_family == "auto":
            return direct_packed_w2a8_linear(
                inputs,
                packed,
                weight_step,
                None,
                n,
                k,
                backend="triton",
                dp4a_config=dp4a_config,
                tensorcore_config=kernel_config,
                activation_backend=activation_backend,
            )
        if kernel_family == "dp4a":
            return direct_packed_dp4a_int8_linear(
                inputs,
                packed,
                weight_step,
                None,
                n,
                k,
                backend="triton",
                kernel_config=dp4a_config,
                activation_backend=activation_backend,
            )
        return direct_packed_int8_linear(
            inputs,
            packed,
            weight_step,
            None,
            n,
            k,
            backend="triton",
            kernel_config=kernel_config,
            activation_backend=activation_backend,
            decode_mode=decode_mode,
        )

    dense_output = dense()
    direct_output = direct()
    torch.cuda.synchronize()
    difference = (direct_output.float() - dense_output.float()).abs()

    dense_samples = _latency_ms(dense, warmup=warmup, iterations=iterations)
    direct_samples = _latency_ms(direct, warmup=warmup, iterations=iterations)
    dense_summary = _summary(dense_samples)
    direct_summary = _summary(direct_samples)
    return {
        "shape_mkn": [m, k, n],
        "kernel_family": kernel_family,
        "activation_backend": activation_backend,
        "decode_mode": decode_mode,
        "kernel_config": (
            None
            if kernel_config is None
            else {
                "block_m": kernel_config.block_m,
                "block_n": kernel_config.block_n,
                "block_k": kernel_config.block_k,
                "num_warps": kernel_config.num_warps,
            }
        ),
        "dp4a_config": (
            None
            if dp4a_config is None
            else {
                "block_n": dp4a_config.block_n,
                "block_groups": dp4a_config.block_groups,
                "num_warps": dp4a_config.num_warps,
            }
        ),
        "dense": dense_summary,
        "direct_w2a8": direct_summary,
        "p50_speedup": dense_summary["p50_ms"] / direct_summary["p50_ms"],
        "storage": {
            "dense_weight_bytes": dense_weight.numel() * dense_weight.element_size(),
            "packed_weight_bytes": packed.numel() * packed.element_size(),
        },
        "strict_bf16_gate": {
            "exact": bool(torch.equal(direct_output, dense_output)),
            "differing_values": int(torch.count_nonzero(difference).item()),
            "total_values": difference.numel(),
            "max_absolute_error": float(difference.max().item()),
            "mean_absolute_error": float(difference.mean().item()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", type=_parse_shape, dest="shapes")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--activation-backend",
        choices=("torch", "hybrid", "triton"),
        default="torch",
    )
    parser.add_argument("--decode-mode", choices=("scalar", "lane4"), default="scalar")
    parser.add_argument(
        "--kernel-family",
        choices=("auto", "tensorcore", "dp4a"),
        default="auto",
    )
    parser.add_argument(
        "--kernel-config",
        action="append",
        type=_parse_kernel_config,
        dest="kernel_configs",
        help="repeat to benchmark multiple BLOCK_M,BLOCK_N,BLOCK_K,WARPS choices",
    )
    parser.add_argument(
        "--dp4a-config",
        action="append",
        type=_parse_dp4a_config,
        dest="dp4a_configs",
        help="repeat to benchmark multiple BLOCK_N,BLOCK_GROUPS,WARPS choices",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    results = {
        "schema_version": 1,
        "device": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "protocol": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timing": "CUDA events; activation quantization plus projection; weights preloaded",
        },
        "measurements": [],
    }
    if args.kernel_family == "auto":
        configs = [
            (tensorcore_config, dp4a_config)
            for tensorcore_config in args.kernel_configs or [None]
            for dp4a_config in args.dp4a_configs or [None]
        ]
    elif args.kernel_family == "dp4a":
        configs = [(None, dp4a_config) for dp4a_config in args.dp4a_configs or [None]]
    else:
        configs = [(tensorcore_config, None) for tensorcore_config in args.kernel_configs or [None]]
    for tensorcore_config, dp4a_config in configs:
        for shape in args.shapes or DEFAULT_SHAPES:
            results["measurements"].append(
                benchmark_shape(
                    shape,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    seed=args.seed,
                    kernel_config=tensorcore_config,
                    dp4a_config=dp4a_config,
                    activation_backend=args.activation_backend,
                    decode_mode=args.decode_mode,
                    kernel_family=args.kernel_family,
                )
            )
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
