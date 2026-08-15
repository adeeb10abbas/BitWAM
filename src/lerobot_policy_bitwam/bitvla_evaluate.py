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
        enable_torch_int8_bitlinear_runtime,
        pack_bitlinear_weights,
    )

    original_initialize = evaluator.initialize_model

    def initialize_packed(cfg: Any) -> tuple[Any, ...]:
        components = original_initialize(cfg)
        model = components[0]
        report = pack_bitlinear_weights(
            model,
            scope=str(config.get("packed_scope", "all")),
        ).to_dict()
        backend = str(config.get("packed_runtime_backend", "compiled_unpack"))
        if backend == "compiled_unpack":
            report["compiled_unpack_layers"] = enable_compiled_bitlinear_unpack(model)
        elif backend == "compiled_projection":
            report["compiled_projection_layers"] = enable_compiled_bitlinear_projection(model)
        elif backend == "compiled":
            report["compiled_layers"] = enable_compiled_bitlinear_runtime(model)
        elif backend == "torch_int8":
            report["torch_int8_layers"] = enable_torch_int8_bitlinear_runtime(model)
        elif backend != "eager_unpack":
            raise ValueError(f"Unsupported packed_runtime_backend: {backend}")
        report["backend"] = backend
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
