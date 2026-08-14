"""Command construction tests for resumable GPU workflows."""

from pathlib import Path

import pytest

from lerobot_policy_bitwam.workflows import build_evaluate_command, build_train_command


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
    assert "--policy.type=bitwam" in command
    assert "--policy.pretrained_path=lerobot/VLA-JEPA-LIBERO" in command
    assert "--policy.quantization_scope=qwen" in command
    assert "--accelerator.mixed_precision=bf16" in command
    assert "--steps=2000" in command


def test_evaluation_distributes_50_episodes_over_ten_tasks() -> None:
    command = build_evaluate_command(_pilot())
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
