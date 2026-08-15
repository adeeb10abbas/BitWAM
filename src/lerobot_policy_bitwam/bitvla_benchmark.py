"""Measure native BitVLA deployment or BitWAM world-head performance on CUDA."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot_policy_bitwam.bitvla_evaluate import _load_upstream_evaluator
from lerobot_policy_bitwam.bitvla_packing import (
    enable_compiled_bitlinear_unpack,
    enable_torch_int8_bitlinear_runtime,
    pack_bitlinear_weights,
)
from lerobot_policy_bitwam.bitvla_world import LatentWorldModelHead
from lerobot_policy_bitwam.workflows import load_config

MIB = 1024**2

_DEPLOYMENT_PATTERNS = (
    "action_head--*_checkpoint.pt",
    "bitvla_for_action_prediction.py",
    "config.json",
    "configuration_bit_vla.py",
    "dataset_statistics.json",
    "generation_config.json",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "proprio_projector--*_checkpoint.pt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required benchmark field: {key}")
    return config[key]


def summarize_latencies(latencies_ms: Sequence[float]) -> dict[str, float | int]:
    """Summarize synchronized latency samples without optional dependencies."""
    if not latencies_ms:
        raise ValueError("at least one latency sample is required")
    ordered = sorted(float(value) for value in latencies_ms)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "iterations": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "stddev_ms": statistics.pstdev(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def deployment_artifacts(checkpoint: Path) -> list[dict[str, int | str]]:
    """List only the files loaded by the native evaluation action path."""
    artifacts: dict[Path, dict[str, int | str]] = {}
    for pattern in _DEPLOYMENT_PATTERNS:
        for path in checkpoint.glob(pattern):
            if path.is_file() and ".back." not in path.name:
                artifacts[path] = {"name": path.name, "bytes": path.stat().st_size}
    return [artifacts[path] for path in sorted(artifacts, key=lambda item: item.name)]


def _module_storage(modules: Iterable[torch.nn.Module | None]) -> dict[str, int]:
    parameter_ids: set[int] = set()
    buffer_ids: set[int] = set()
    parameter_count = parameter_bytes = buffer_bytes = 0
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if id(parameter) in parameter_ids:
                continue
            parameter_ids.add(id(parameter))
            parameter_count += parameter.numel()
            parameter_bytes += parameter.numel() * parameter.element_size()
        for buffer in module.buffers():
            if id(buffer) in buffer_ids:
                continue
            buffer_ids.add(id(buffer))
            buffer_bytes += buffer.numel() * buffer.element_size()
    return {
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
    }


def _device_metadata() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }


def _fresh_observation(images: tuple[np.ndarray, np.ndarray], state: np.ndarray) -> dict[str, Any]:
    return {
        "full_image": images[0],
        "wrist_image": images[1],
        "state": state.copy(),
    }


def _benchmark_policy(config: dict[str, Any]) -> dict[str, Any]:
    evaluator = _load_upstream_evaluator(config)
    checkpoint = Path(_required(config, "checkpoint"))
    warmup = int(config.get("benchmark_warmup_iterations", 10))
    iterations = int(config.get("benchmark_iterations", 50))
    if warmup < 1 or iterations < 1:
        raise ValueError("benchmark warmup and timed iterations must be positive")

    evaluator.set_seed_everywhere(int(config.get("seed", 0)))
    cfg = evaluator.GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=str(config.get("task_suite_name", "libero_10")),
        model_family="bitnet",
        use_wandb=False,
        seed=int(config.get("seed", 0)),
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, _ = torch.cuda.mem_get_info()
    load_started = time.perf_counter()
    model, action_head, proprio_projector, noisy_action_projector, processor = (
        evaluator.initialize_model(cfg)
    )
    packing = None
    if bool(config.get("packed_runtime", False)):
        packing = pack_bitlinear_weights(
            model,
            scope=str(config.get("packed_scope", "all")),
        ).to_dict()
        packed_backend = str(config.get("packed_runtime_backend", "eager_unpack"))
        if packed_backend == "compiled_unpack":
            packing["compiled_unpack_layers"] = enable_compiled_bitlinear_unpack(model)
        elif packed_backend == "torch_int8":
            packing["torch_int8_layers"] = enable_torch_int8_bitlinear_runtime(model)
        elif packed_backend != "eager_unpack":
            raise ValueError(f"Unsupported packed_runtime_backend: {packed_backend}")
        packing["backend"] = packed_backend
        gc.collect()
        torch.cuda.empty_cache()
    model.set_constant(
        image_token_idx=evaluator.BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        proprio_pad_idx=evaluator.BITNET_PROPRIO_PAD_IDX,
        ignore_idx=evaluator.BITNET_IGNORE_INDEX,
        action_token_begin_idx=evaluator.BITNET_ACTION_TOKEN_BEGIN_IDX,
        stop_index=evaluator.BITNET_STOP_INDEX,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    free_after_load, _ = torch.cuda.mem_get_info()
    allocated_after_load = torch.cuda.memory_allocated()
    reserved_after_load = torch.cuda.memory_reserved()
    load_peak_allocated = torch.cuda.max_memory_allocated()

    rng = np.random.default_rng(int(config.get("seed", 0)))
    images = (
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
    )
    state = np.zeros(8, dtype=np.float32)
    task_label = str(config.get("benchmark_task_label", "put the black bowl on the plate"))

    def query() -> None:
        evaluator.get_action(
            cfg,
            model,
            _fresh_observation(images, state),
            task_label,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
        )

    for _ in range(warmup):
        query()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies_ms: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        query()
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter_ns() - started) / 1e6)

    latency = summarize_latencies(latencies_ms)
    latency["action_chunks_per_second_at_p50"] = 1000 / float(latency["p50_ms"])
    latency["control_steps_per_second_at_p50"] = 8000 / float(latency["p50_ms"])
    artifacts = deployment_artifacts(checkpoint)
    world_files = sorted(checkpoint.glob("world_model--*_checkpoint.pt"))
    optimizer_files = sorted(checkpoint.glob("optimizer--*_checkpoint.pt"))
    return {
        "schema_version": 1,
        "target": "native_bitvla_policy_query",
        "run_id": str(_required(config, "run_id")),
        "checkpoint": str(checkpoint),
        "device": _device_metadata(),
        "protocol": {
            "batch_size": 1,
            "action_chunk_size": int(cfg.num_open_loop_steps),
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "timing_scope": (
                "official get_action, including input preprocessing and synchronized CUDA inference"
            ),
            "input": "fixed seeded 224x224 RGB main/wrist images and zero proprioception",
            "packed_runtime": packing is not None,
        },
        "packing": packing,
        "load_seconds": load_seconds,
        "latency": latency,
        "latency_samples_ms": latencies_ms,
        "memory": {
            "parameter_and_buffer_storage": _module_storage(
                (model, action_head, proprio_projector, noisy_action_projector)
            ),
            "cuda_allocated_after_load_bytes": allocated_after_load,
            "cuda_reserved_after_load_bytes": reserved_after_load,
            "cuda_device_usage_delta_after_load_bytes": free_before - free_after_load,
            "cuda_peak_allocated_during_load_bytes": load_peak_allocated,
            "cuda_peak_allocated_during_query_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_query_increment_bytes": max(
                0, torch.cuda.max_memory_allocated() - allocated_after_load
            ),
        },
        "artifacts": {
            "deployment_files": artifacts,
            "deployment_bytes": sum(int(item["bytes"]) for item in artifacts),
            "world_model_bytes_excluded_from_deployment": sum(path.stat().st_size for path in world_files),
            "optimizer_bytes_excluded_from_deployment": sum(path.stat().st_size for path in optimizer_files),
        },
    }


def _world_head_step(
    head: LatentWorldModelHead,
    hidden: torch.Tensor,
    actions: torch.Tensor,
    target: torch.Tensor,
) -> None:
    head.zero_grad(set_to_none=True)
    output = head(
        hidden,
        actions,
        target,
        shuffled_actions=actions.roll(1, dims=0),
        contrastive_margin=0.05,
    )
    loss = output.loss
    if output.contrastive_loss is not None:
        loss = loss + 0.1 * output.contrastive_loss
    loss.backward()


def _benchmark_world_head(config: dict[str, Any]) -> dict[str, Any]:
    warmup = int(config.get("benchmark_warmup_iterations", 10))
    iterations = int(config.get("benchmark_iterations", 50))
    batch_size = int(config.get("benchmark_batch_size", 8))
    ternary = bool(_required(config, "world_head_ternary"))
    latent_dim = int(config.get("world_head_latent_dim", 2560))
    hidden_dim = int(config.get("world_head_hidden_dim", 2048))
    if min(warmup, iterations, batch_size) < 1:
        raise ValueError("benchmark warmup, timed iterations, and batch size must be positive")

    torch.manual_seed(int(config.get("seed", 0)))
    torch.cuda.manual_seed_all(int(config.get("seed", 0)))
    gc.collect()
    torch.cuda.empty_cache()
    head = LatentWorldModelHead(
        latent_dim,
        action_chunk_size=8,
        action_dim=7,
        hidden_dim=hidden_dim,
        ternary=ternary,
    ).to(device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(batch_size, 8, latent_dim, device="cuda", dtype=torch.bfloat16)
    actions = torch.randn(batch_size, 8, 7, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(batch_size, latent_dim, device="cuda", dtype=torch.bfloat16)

    for _ in range(warmup):
        _world_head_step(head, hidden, actions, target)
    torch.cuda.synchronize()
    allocated_after_load = torch.cuda.memory_allocated()
    reserved_after_load = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    latencies_ms: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        _world_head_step(head, hidden, actions, target)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter_ns() - started) / 1e6)

    storage = _module_storage((head,))
    eligible_weights = sum(
        module.weight.numel() for module in head.modules() if isinstance(module, torch.nn.Linear)
    )
    return {
        "schema_version": 1,
        "target": "bitwam_world_head_training_step",
        "run_id": str(_required(config, "run_id")),
        "variant": "ternary_qat" if ternary else "bf16",
        "device": _device_metadata(),
        "protocol": {
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "action_tokens": 8,
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "timing_scope": (
                "world loss plus shuffled-action contrastive forward and backward; optimizer excluded"
            ),
        },
        "latency": summarize_latencies(latencies_ms),
        "latency_samples_ms": latencies_ms,
        "memory": {
            "parameter_and_buffer_storage": storage,
            "cuda_allocated_after_warmup_bytes": allocated_after_load,
            "cuda_reserved_after_warmup_bytes": reserved_after_load,
            "cuda_peak_allocated_during_step_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_step_increment_bytes": max(
                0, torch.cuda.max_memory_allocated() - allocated_after_load
            ),
        },
        "packing": {
            "eligible_ternary_weight_count": eligible_weights,
            "current_master_weight_bytes": eligible_weights * 2,
            "theoretical_two_bit_weight_bytes": math.ceil(eligible_weights * 2 / 8),
            "theoretical_entropy_1_58_bit_weight_bytes": math.ceil(eligible_weights * math.log2(3) / 8),
            "packed_kernel_implemented": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if config.get("architecture") != "bitvla":
        raise ValueError("BitVLA benchmark configs must set architecture: bitvla")
    if not torch.cuda.is_available():
        raise RuntimeError("The native BitVLA benchmark requires CUDA")
    target = str(config.get("benchmark_target", "policy"))
    if target == "policy":
        result = _benchmark_policy(config)
    elif target == "world_head":
        result = _benchmark_world_head(config)
    else:
        raise ValueError(f"Unsupported benchmark_target: {target}")

    output_dir = Path(_required(config, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_metrics.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output_path), **result["latency"]}, indent=2))


if __name__ == "__main__":
    main()
