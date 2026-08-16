"""Closed-loop LIBERO evaluation through BitVLA's pinned official harness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from lerobot_policy_bitwam.workflows import load_config


def _enable_packed_runtime(evaluator: ModuleType, config: dict[str, Any]) -> None:
    """Wrap upstream model initialization with BitWAM's packed inference conversion."""
    from lerobot_policy_bitwam.bitvla_packing import (
        enable_compiled_bitlinear_projection,
        enable_compiled_bitlinear_runtime,
        enable_compiled_bitlinear_unpack,
        enable_direct_w2a8_bitlinear_runtime,
        enable_torch_int8_bitlinear_runtime,
        enable_triton_direct_bitlinear_runtime,
        pack_bitlinear_weights,
    )

    original_initialize = evaluator.initialize_model

    def initialize_packed(cfg: Any) -> tuple[Any, ...]:
        backend = str(config.get("packed_runtime_backend", "compiled_unpack"))
        artifact = config.get("packed_artifact")
        if artifact is not None:
            if backend != "triton_w2a8":
                raise ValueError("Direct packed artifacts currently require triton_w2a8")
            if not hasattr(evaluator, "get_model"):
                raise RuntimeError("Upstream evaluator does not expose its get_model loader")
            from lerobot_policy_bitwam.bitvla_packed_checkpoint import (
                build_bitvla_topology,
                load_packed_checkpoint,
                move_packed_bitvla_to_device,
            )

            source_checkpoint = Path(_required(config, "checkpoint")).expanduser().resolve()
            artifact_path = Path(artifact).expanduser().resolve()
            original_get_model = evaluator.get_model
            loaded: dict[str, Any] = {}

            def get_packed_model(_cfg: Any) -> Any:
                model = build_bitvla_topology(source_checkpoint)
                manifest = load_packed_checkpoint(
                    model,
                    artifact_path,
                    expected_source_metadata={"checkpoint": str(source_checkpoint)},
                )
                if manifest.packing.get("packed_layout", "row_major") != "dp4a":
                    raise RuntimeError("Production W2A8 artifacts must use the DP4A layout")
                statistics_path = source_checkpoint / "dataset_statistics.json"
                if not statistics_path.is_file():
                    raise RuntimeError(
                        "Direct packed evaluation requires dataset_statistics.json beside "
                        "the source checkpoint"
                    )
                norm_stats = json.loads(statistics_path.read_text(encoding="utf-8"))
                if not isinstance(norm_stats, dict) or not norm_stats:
                    raise RuntimeError("Packed evaluation dataset statistics are invalid")
                model.norm_stats = norm_stats
                loaded["manifest"] = manifest
                return move_packed_bitvla_to_device(
                    model,
                    str(config.get("packed_device", "cuda:0")),
                )

            evaluator.get_model = get_packed_model
            try:
                components = original_initialize(cfg)
            finally:
                evaluator.get_model = original_get_model
            manifest = loaded.get("manifest")
            if manifest is None:
                raise RuntimeError("Upstream initialization did not invoke the packed model loader")
            model = components[0]
            report = manifest.packing
            report["packed_layers"] = len(manifest.packed_layers)
            report["artifact"] = str(artifact_path)
            report["artifact_bytes"] = sum(
                path.stat().st_size for path in artifact_path.rglob("*") if path.is_file()
            )
            report["direct_artifact_load"] = True
        else:
            components = original_initialize(cfg)
            model = components[0]
            report = pack_bitlinear_weights(
                model,
                scope=str(config.get("packed_scope", "all")),
            ).to_dict()
        if backend == "compiled_unpack":
            report["compiled_unpack_layers"] = enable_compiled_bitlinear_unpack(model)
        elif backend == "compiled_projection":
            report["compiled_projection_layers"] = enable_compiled_bitlinear_projection(model)
        elif backend == "compiled":
            report["compiled_layers"] = enable_compiled_bitlinear_runtime(model)
        elif backend == "torch_int8":
            report["torch_int8_layers"] = enable_torch_int8_bitlinear_runtime(model)
        elif backend == "triton_w2a8":
            direct_activation = str(config.get("packed_activation_backend", "hybrid"))
            small_batch_threshold = int(config.get("packed_small_batch_threshold", 8))
            report["triton_w2a8_layers"] = enable_direct_w2a8_bitlinear_runtime(
                model,
                activation_backend=direct_activation,
                small_batch_threshold=small_batch_threshold,
            )
            report["activation_backend"] = direct_activation
            report["small_batch_threshold"] = small_batch_threshold
            report["packed_layout"] = "dp4a"
        elif backend in {"triton_direct_int8", "triton_direct_bf16"}:
            direct_activation = str(config.get("packed_activation_backend", "torch"))
            report["triton_direct_layers"] = enable_triton_direct_bitlinear_runtime(
                model,
                activation_backend=direct_activation,
                bf16_candidate=backend == "triton_direct_bf16",
            )
            report["activation_backend"] = direct_activation
        elif backend != "eager_unpack":
            raise ValueError(f"Unsupported packed_runtime_backend: {backend}")
        report["backend"] = backend
        evaluator._bitwam_packing_report = dict(report)
        print("BitWAM packed runtime: " + json.dumps(report, sort_keys=True))
        return components

    evaluator.initialize_model = initialize_packed


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required BitVLA evaluation field: {key}")
    return config[key]


def build_upstream_eval_argv(config: dict[str, Any]) -> list[str]:
    """Translate the stable BitWAM evaluation schema to BitVLA's Draccus CLI."""
    episodes = int(config.get("episodes", 10))
    task_count = int(config.get("task_count", 10))
    if episodes < 1 or task_count < 1 or episodes % task_count:
        raise ValueError("episodes must be positive and divide evenly across task_count")

    checkpoint = _required(config, "checkpoint")
    output_dir = _required(config, "output_dir")
    run_id = str(_required(config, "run_id"))
    values = {
        "pretrained_checkpoint": checkpoint,
        "task_suite_name": config.get("task_suite_name", "libero_10"),
        "num_trials_per_task": episodes // task_count,
        "model_family": "bitnet",
        "use_wandb": False,
        "local_log_dir": output_dir,
        "info_in_path": run_id,
        "run_id_note": run_id,
        "seed": int(config.get("seed", 7)),
    }
    argv = ["run_libero_eval_bitnet.py"]
    for key, value in values.items():
        if isinstance(value, bool):
            value = "True" if value else "False"
        argv.extend((f"--{key}", str(value)))
    return argv


def _load_upstream_evaluator(config: dict[str, Any]) -> ModuleType:
    upstream_root = Path(_required(config, "upstream_root")).expanduser().resolve()
    expected_revision = str(_required(config, "upstream_revision"))
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected_revision:
        raise ValueError(f"BitVLA revision mismatch: expected {expected_revision}, found {revision}")

    openvla_root = upstream_root / "openvla-oft"
    libero_root = upstream_root / "LIBERO"
    script = openvla_root / "experiments/robot/libero/run_libero_eval_bitnet.py"
    if not script.is_file() or not libero_root.is_dir():
        raise ValueError(f"BitVLA evaluation checkout is incomplete: {upstream_root}")
    for path in (
        str(openvla_root / "bitvla"),
        str(libero_root),
        str(openvla_root),
        str(upstream_root),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.chdir(openvla_root)
    spec = importlib.util.spec_from_file_location("bitwam_upstream_bitvla_eval", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load upstream evaluator: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if config.get("architecture") != "bitvla":
        raise ValueError("BitVLA evaluation configs must set architecture: bitvla")
    evaluator = _load_upstream_evaluator(config)
    if bool(config.get("packed_runtime", False)):
        _enable_packed_runtime(evaluator, config)
    sys.argv = build_upstream_eval_argv(config)
    evaluator.eval_libero()


if __name__ == "__main__":
    main()
