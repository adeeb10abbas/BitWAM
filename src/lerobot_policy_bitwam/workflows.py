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


def _train_launcher(num_processes: int) -> list[str]:
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    if num_processes == 1:
        return [sys.executable, "-m", "lerobot_policy_bitwam.train_entrypoint"]
    torchrun = Path(sys.executable).with_name("torchrun")
    return [
        str(torchrun),
        "--standalone",
        f"--nproc-per-node={num_processes}",
        "--module",
        "lerobot_policy_bitwam.train_entrypoint",
    ]


def _bitvla_train_command(config: dict[str, Any]) -> tuple[str, ...]:
    config_path = _required(config, "_config_path")
    num_processes = int(config.get("num_processes", 1))
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    torchrun = Path(sys.executable).with_name("torchrun")
    return (
        str(torchrun),
        "--standalone",
        f"--nproc-per-node={num_processes}",
        "--module",
        "lerobot_policy_bitwam.bitvla_train",
        "--config",
        str(config_path),
    )


def _resume_config(output_dir: Path) -> Path | None:
    candidate = _latest_checkpoint(output_dir) / "train_config.json"
    state_path, _ = _metadata_paths(output_dir, "train")
    if not candidate.is_file():
        return None
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "completed":
            return None
    return candidate


def _latest_checkpoint(output_dir: Path) -> Path:
    """Return the most recent saved policy, including LeRobot's numeric-only layout."""
    checkpoints = output_dir / "checkpoints"
    candidates = [
        path
        for path in checkpoints.glob("*")
        if path.name.isdigit() and (path / "pretrained_model").is_dir()
    ]
    if candidates:
        return max(candidates, key=lambda path: int(path.name)) / "pretrained_model"
    return checkpoints / "last" / "pretrained_model"


def _is_bf16_materialization(config: dict[str, Any]) -> bool:
    return (
        config.get("quantization_scope", "none") == "none"
        and float(config.get("learning_rate", 1e-4)) == 0.0
        and int(config.get("steps", 0)) == 1
    )


def _materialization_checkpoint(config: dict[str, Any]) -> Path:
    if checkpoint := config.get("checkpoint"):
        return Path(checkpoint)
    return Path(_required(config, "output_dir")) / "checkpoints" / "000001" / "pretrained_model"


def build_train_command(config: dict[str, Any]) -> tuple[str, ...]:
    """Translate a BitWAM experiment config to the pinned LeRobot trainer."""
    if config.get("architecture") == "bitvla":
        return _bitvla_train_command(config)
    source = _required(config, "source_checkpoint")
    output_dir = Path(_required(config, "output_dir"))
    scope = config.get("quantization_scope", "none")
    world_weight = float(config.get("world_loss_weight", 0.1))
    num_processes = int(config.get("num_processes", 1))
    if _is_bf16_materialization(config):
        return tuple(
            option
            for option in (
                sys.executable,
                "-m",
                "lerobot_policy_bitwam.materialize",
                _option("source-checkpoint", source),
                _option("output-checkpoint", _materialization_checkpoint(config)),
                _option("source-revision", config.get("source_revision")),
                _option("world-loss-weight", world_weight),
            )
            if not option.endswith("=None")
        )
    command = _train_launcher(num_processes)
    if resume_config := _resume_config(output_dir):
        command.extend(
            [
                _option("discover_packages_path", "lerobot_policy_bitwam"),
                _option("resume", True),
                _option("config_path", resume_config),
            ]
        )
        return tuple(command)

    command.extend(
        [
            _option("discover_packages_path", "lerobot_policy_bitwam"),
            _option("policy.type", "bitwam"),
            _option("policy.pretrained_path", source),
            _option("policy.source_checkpoint", source),
            _option("policy.source_revision", config.get("source_revision")),
            _option("policy.world_loss_weight", world_weight),
            _option("policy.world_model_loss_weight", world_weight),
            _option("policy.quantization_scope", scope),
            _option("policy.qat_recovery", config.get("qat_recovery", "none")),
            _option(
                "policy.representation_distillation_weight",
                config.get("representation_distillation_weight", 0.0),
            ),
            _option("policy.inference_backend", "native"),
            _option("policy.optimizer_lr", config.get("learning_rate", 1e-4)),
            _option("policy.device", "cuda"),
            _option("policy.torch_dtype", "bfloat16"),
            _option("policy.push_to_hub", False),
            _option("dataset.repo_id", _required(config, "dataset_repo")),
            _option("dataset.streaming", config.get("dataset_streaming", False)),
            _option("output_dir", output_dir),
            _option("job_name", _required(config, "run_id")),
            _option("steps", _required(config, "steps")),
            _option("batch_size", config.get("batch_size", 8)),
            _option("num_workers", config.get("num_workers", 1)),
            _option("seed", _required(config, "seed")),
            _option("save_freq", config.get("save_freq", 500)),
            _option("save_checkpoint", config.get("save_checkpoint", True)),
            _option("log_freq", config.get("log_freq", 20)),
            _option("accelerator.mixed_precision", "bf16"),
            _option(
                "accelerator.gradient_accumulation.steps",
                config.get("gradient_accumulation_steps", 1),
            ),
        ]
    )
    if num_processes > 1:
        command.extend(
            [
                _option("parallelism.dp_shard", num_processes),
                _option("accelerator.fsdp.reshard_after_forward", True),
                _option(
                    "accelerator.fsdp.cpu_offload",
                    config.get("fsdp_cpu_offload", False),
                ),
            ]
        )
    return tuple(option for option in command if not option.endswith("=None"))


def _evaluation_checkpoint(config: dict[str, Any]) -> str:
    if checkpoint := config.get("checkpoint"):
        return str(checkpoint)
    if config.get("quantization_scope", "none") == "none":
        return str(_required(config, "source_checkpoint"))
    return str(_latest_checkpoint(Path(_required(config, "output_dir"))))


def build_evaluate_command(config: dict[str, Any]) -> tuple[str, ...]:
    """Translate a config to a 50-episode LIBERO-10 evaluation command."""
    episodes = int(config.get("episodes", 50))
    task_count = int(config.get("task_count", 10))
    if episodes % task_count:
        raise ValueError("episodes must divide evenly across LIBERO-10 tasks")
    episodes_per_task = episodes // task_count
    output_dir = Path(config.get("_evaluation_output_dir", _required(config, "output_dir"))) / "evaluation"
    return (
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        _option("discover_packages_path", "lerobot_policy_bitwam"),
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


def _metadata_paths(output_dir: Path, stage: str) -> tuple[Path, Path]:
    """Return metadata paths without pre-creating LeRobot's guarded training directory."""
    if stage == "train":
        prefix = output_dir.parent / f".{output_dir.name}"
        return (
            prefix.with_name(f"{prefix.name}.{stage}_state.json"),
            prefix.with_name(f"{prefix.name}.{stage}.log"),
        )
    return output_dir / f"{stage}_state.json", output_dir / f"{stage}.log"


def _run_external(command: Sequence[str], output_dir: Path, stage: str) -> int:
    """Run a long job with a PID, durable log, exact command, and safe completed-run skip."""
    if stage == "train":
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    state_path, log_path = _metadata_paths(output_dir, stage)
    command_id = _command_id(command)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "completed" and state.get("command_id") == command_id:
            print(f"Skipping completed {stage} job recorded in {state_path}")
            return 0

    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
    if command == "screen":
        screen_config = config | {
            "episodes": int(config.get("screen_episodes", 10)),
            "_evaluation_output_dir": str(output_dir / "screen-1000"),
        }
        screen_output_dir = Path(screen_config["_evaluation_output_dir"])
        return _run_external(build_evaluate_command(screen_config), screen_output_dir, "screen")
    if command == "evaluate":
        return _run_external(build_evaluate_command(config), output_dir, "evaluate")
    raise NotImplementedError(f"The {command!r} workflow is introduced by a later execution phase.")
