"""Resumable workflow dispatch used by the public CLI."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML mapping and retain the exact source path for provenance."""
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    config["_config_path"] = str(config_path.resolve())
    return config


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required configuration field: {key}")
    return config[key]


def _option(name: str, value: Any) -> str:
    if isinstance(value, bool):
        serialized = str(value).lower()
    elif isinstance(value, (list, dict)):
        serialized = json.dumps(value, separators=(",", ":"))
    else:
        serialized = str(value)
    return f"--{name}={serialized}"


def build_train_command(config: dict[str, Any]) -> tuple[str, ...]:
    """Translate a BitWAM experiment config to the pinned LeRobot trainer."""
    source = _required(config, "source_checkpoint")
    output_dir = _required(config, "output_dir")
    scope = config.get("quantization_scope", "none")
    world_weight = float(config.get("world_loss_weight", 0.1))
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        _option("policy.discover_packages_path", "lerobot_policy_bitwam"),
        _option("policy.type", "bitwam"),
        _option("policy.pretrained_path", source),
        _option("policy.source_checkpoint", source),
        _option("policy.source_revision", config.get("source_revision")),
        _option("policy.world_loss_weight", world_weight),
        _option("policy.world_model_loss_weight", world_weight),
        _option("policy.quantization_scope", scope),
        _option("policy.qat_recovery", config.get("qat_recovery", "none")),
        _option("policy.inference_backend", "native"),
        _option("policy.optimizer_lr", config.get("learning_rate", 1e-4)),
        _option("policy.device", "cuda"),
        _option("policy.torch_dtype", "bfloat16"),
        _option("policy.push_to_hub", False),
        _option("dataset.repo_id", _required(config, "dataset_repo")),
        _option("output_dir", output_dir),
        _option("job_name", _required(config, "run_id")),
        _option("steps", _required(config, "steps")),
        _option("batch_size", config.get("batch_size", 8)),
        _option("num_workers", config.get("num_workers", 4)),
        _option("seed", _required(config, "seed")),
        _option("save_freq", config.get("save_freq", 500)),
        _option("log_freq", config.get("log_freq", 20)),
        _option("accelerator.mixed_precision", "bf16"),
        _option(
            "accelerator.gradient_accumulation.steps",
            config.get("gradient_accumulation_steps", 1),
        ),
    ]
    return tuple(option for option in command if not option.endswith("=None"))


def _evaluation_checkpoint(config: dict[str, Any]) -> str:
    if checkpoint := config.get("checkpoint"):
        return str(checkpoint)
    if config.get("quantization_scope", "none") == "none":
        return str(_required(config, "source_checkpoint"))
    return str(Path(_required(config, "output_dir")) / "checkpoints" / "last" / "pretrained_model")


def build_evaluate_command(config: dict[str, Any]) -> tuple[str, ...]:
    """Translate a config to a 50-episode LIBERO-10 evaluation command."""
    episodes = int(config.get("episodes", 50))
    task_count = int(config.get("task_count", 10))
    if episodes % task_count:
        raise ValueError("episodes must divide evenly across LIBERO-10 tasks")
    episodes_per_task = episodes // task_count
    output_dir = Path(_required(config, "output_dir")) / "evaluation"
    return (
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        _option("policy.discover_packages_path", "lerobot_policy_bitwam"),
        _option("policy.path", _evaluation_checkpoint(config)),
        _option("policy.device", "cuda"),
        _option("env.type", "libero"),
        _option("env.task", "libero_10"),
        _option("env.max_parallel_tasks", 1),
        _option("eval.n_episodes", episodes_per_task),
        _option("eval.batch_size", config.get("eval_batch_size", 1)),
        _option("seed", _required(config, "seed")),
        _option("output_dir", output_dir),
        _option("job_name", f"{_required(config, 'run_id')}-eval"),
    )


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _command_id(command: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(command)).encode()).hexdigest()


def _run_external(command: Sequence[str], output_dir: Path, stage: str) -> int:
    """Run a long job with a PID, durable log, exact command, and safe completed-run skip."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{stage}_state.json"
    log_path = output_dir / f"{stage}.log"
    command_id = _command_id(command)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "completed" and state.get("command_id") == command_id:
            print(f"Skipping completed {stage} job recorded in {state_path}")
            return 0

    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        _write_state(
            state_path,
            {
                "status": "running",
                "pid": process.pid,
                "command": list(command),
                "command_id": command_id,
                "log": str(log_path.resolve()),
            },
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="")
        return_code = process.wait()

    _write_state(
        state_path,
        {
            "status": "completed" if return_code == 0 else "failed",
            "pid": process.pid,
            "return_code": return_code,
            "command": list(command),
            "command_id": command_id,
            "log": str(log_path.resolve()),
        },
    )
    return return_code


def run_command(command: str, config_path: Path) -> int:
    """Dispatch one CLI workflow while keeping GPU jobs resumable and inspectable."""
    config = load_config(config_path)
    output_dir = Path(_required(config, "output_dir"))
    if command == "train":
        return _run_external(build_train_command(config), output_dir, "train")
    if command == "evaluate":
        return _run_external(build_evaluate_command(config), output_dir, "evaluate")
    raise NotImplementedError(f"The {command!r} workflow is introduced by a later execution phase.")
