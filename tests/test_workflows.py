"""Command construction tests for resumable GPU workflows."""

from pathlib import Path

import pytest

from lerobot_policy_bitwam.workflows import (
    _metadata_paths,
    build_evaluate_command,
    build_train_command,
)


def _pilot() -> dict:
    return {
        "run_id": "pilot",
        "seed": 0,
        "source_checkpoint": "lerobot/VLA-JEPA-LIBERO",
        "dataset_repo": "HuggingFaceVLA/libero",
        "output_dir": "outputs/pilot",
        "steps": 2000,
        "episodes": 50,
        "task_count": 10,
        "quantization_scope": "qwen",
        "world_loss_weight": 0.1,
    }


def test_train_command_uses_plugin_source_and_bf16() -> None:
    command = build_train_command(_pilot())
    assert "--discover_packages_path=lerobot_policy_bitwam" in command
    assert "--policy.type=bitwam" in command
    assert "--policy.pretrained_path=lerobot/VLA-JEPA-LIBERO" in command
    assert "--policy.quantization_scope=qwen" in command
    assert "--accelerator.mixed_precision=bf16" in command
    assert "--save_checkpoint=true" in command
    assert "--steps=2000" in command


def test_two_gpu_training_uses_local_torchrun_and_fsdp() -> None:
    command = build_train_command(
        _pilot()
        | {
            "num_processes": 2,
            "dataset_streaming": True,
            "fsdp_cpu_offload": True,
        }
    )
    assert command[0].endswith("/torchrun")
    assert "--standalone" in command
    assert "--nproc-per-node=2" in command
    assert "--module" in command
    assert "lerobot_policy_bitwam.train_entrypoint" in command
    assert "--parallelism.dp_shard=2" in command
    assert "--accelerator.fsdp.reshard_after_forward=true" in command
    assert "--accelerator.fsdp.cpu_offload=true" in command
    assert "--dataset.streaming=true" in command


def test_training_rejects_zero_processes() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_train_command(_pilot() | {"num_processes": 0})


def test_failed_training_resumes_from_last_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "pilot"
    resume_config = output_dir / "checkpoints/last/pretrained_model/train_config.json"
    resume_config.parent.mkdir(parents=True)
    resume_config.write_text("{}", encoding="utf-8")
    command = build_train_command(_pilot() | {"output_dir": str(output_dir), "num_processes": 2})
    assert "--resume=true" in command
    assert f"--config_path={resume_config}" in command
    assert "--policy.type=bitwam" not in command


def test_training_metadata_does_not_create_a_trainer_output_collision() -> None:
    state, log = _metadata_paths(Path("outputs/pilot"), "train")
    assert state == Path("outputs/.pilot.train_state.json")
    assert log == Path("outputs/.pilot.train.log")


def test_evaluation_metadata_stays_with_the_run() -> None:
    state, log = _metadata_paths(Path("outputs/pilot"), "evaluate")
    assert state == Path("outputs/pilot/evaluate_state.json")
    assert log == Path("outputs/pilot/evaluate.log")


def test_evaluation_distributes_50_episodes_over_ten_tasks() -> None:
    command = build_evaluate_command(_pilot())
    assert "--discover_packages_path=lerobot_policy_bitwam" in command
    assert "--env.task=libero_10" in command
    assert "--eval.n_episodes=5" in command
    assert "--policy.path=outputs/pilot/checkpoints/last/pretrained_model" in command
    assert "--output_dir=outputs/pilot/evaluation" in command


def test_evaluation_rejects_uneven_episode_distribution() -> None:
    config = _pilot() | {"episodes": 51}
    with pytest.raises(ValueError, match="divide evenly"):
        build_evaluate_command(config)


def test_baseline_evaluates_source_checkpoint() -> None:
    config = _pilot() | {"quantization_scope": "none", "output_dir": Path("outputs/baseline")}
    command = build_evaluate_command(config)
    assert "--policy.path=lerobot/VLA-JEPA-LIBERO" in command
