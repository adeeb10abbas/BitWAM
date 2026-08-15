"""Command construction tests for resumable GPU workflows."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot_policy_bitwam.workflows import (
    _metadata_paths,
    build_benchmark_command,
    build_evaluate_command,
    build_train_command,
    run_command,
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
    assert "--policy.representation_distillation_weight=0.0" in command
    assert "--accelerator.mixed_precision=bf16" in command
    assert "--save_checkpoint=true" in command
    assert "--num_workers=1" in command
    assert "--steps=2000" in command


def test_train_command_enables_representation_distillation() -> None:
    command = build_train_command(_pilot() | {"representation_distillation_weight": 0.1})
    assert "--policy.representation_distillation_weight=0.1" in command


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


def test_native_bitvla_training_uses_staged_integration_module(tmp_path: Path) -> None:
    config_path = tmp_path / "bitvla.yaml"
    config_path.write_text("architecture: bitvla\n", encoding="utf-8")
    command = build_train_command(
        {
            "architecture": "bitvla",
            "_config_path": str(config_path),
            "num_processes": 2,
        }
    )
    assert "--nproc-per-node=2" in command
    assert "lerobot_policy_bitwam.bitvla_train" in command
    assert command[-2:] == ("--config", str(config_path))


def test_single_gpu_bitvla_training_still_initializes_torch_distributed(tmp_path: Path) -> None:
    config_path = tmp_path / "bitvla.yaml"
    config_path.write_text("architecture: bitvla\n", encoding="utf-8")
    command = build_train_command(
        {
            "architecture": "bitvla",
            "_config_path": str(config_path),
            "num_processes": 1,
        }
    )
    assert command[0].endswith("/torchrun")
    assert "--nproc-per-node=1" in command


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


def test_evaluation_uses_the_newest_numbered_checkpoint_when_available(tmp_path: Path) -> None:
    output_dir = tmp_path / "pilot"
    for step in (500, 1000):
        (output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model").mkdir(parents=True)
    command = build_evaluate_command(_pilot() | {"output_dir": output_dir})
    assert f"--policy.path={output_dir}/checkpoints/001000/pretrained_model" in command


def test_evaluation_rejects_uneven_episode_distribution() -> None:
    config = _pilot() | {"episodes": 51}
    with pytest.raises(ValueError, match="divide evenly"):
        build_evaluate_command(config)


def test_baseline_evaluates_source_checkpoint() -> None:
    config = _pilot() | {"quantization_scope": "none", "output_dir": Path("outputs/baseline")}
    command = build_evaluate_command(config)
    assert "--policy.path=lerobot/VLA-JEPA-LIBERO" in command


def test_zero_learning_rate_bf16_control_materializes_without_a_scheduler() -> None:
    config = _pilot() | {
        "quantization_scope": "none",
        "learning_rate": 0.0,
        "steps": 1,
        "checkpoint": "outputs/bf16-control/checkpoints/000001/pretrained_model",
    }
    command = build_train_command(config)
    assert command[:3] == (command[0], "-m", "lerobot_policy_bitwam.materialize")
    assert "--source-checkpoint=lerobot/VLA-JEPA-LIBERO" in command
    assert "--output-checkpoint=outputs/bf16-control/checkpoints/000001/pretrained_model" in command
    assert "--world-loss-weight=0.1" in command


def test_materialized_bf16_control_keeps_source_processor_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    from lerobot_policy_bitwam import materialize

    source = tmp_path / "source"
    source.mkdir()
    (source / "policy_preprocessor.json").write_text("pre", encoding="utf-8")
    (source / "policy_postprocessor.json").write_text("post", encoding="utf-8")
    (source / "policy_preprocessor_state.safetensors").write_text("state", encoding="utf-8")
    (source / "not-a-processor.txt").write_text("ignore", encoding="utf-8")
    target = tmp_path / "target"

    class FakePolicy:
        quantization_report = SimpleNamespace(ternary_parameter_count=0)

        @classmethod
        def from_source_checkpoint(cls, checkpoint, **kwargs):
            assert checkpoint == str(source)
            assert kwargs["config_overrides"]["quantization_scope"] == "none"
            return cls()

        def save_pretrained(self, output_checkpoint):
            Path(output_checkpoint).mkdir(parents=True)
            (Path(output_checkpoint) / "model.safetensors").write_text("model", encoding="utf-8")

    monkeypatch.setattr(materialize, "BitWAMPolicy", FakePolicy)
    materialize.materialize_bf16_control(str(source), target)

    assert (target / "model.safetensors").is_file()
    assert (target / "policy_preprocessor.json").read_text(encoding="utf-8") == "pre"
    assert (target / "policy_postprocessor.json").read_text(encoding="utf-8") == "post"
    assert (target / "policy_preprocessor_state.safetensors").read_text(encoding="utf-8") == "state"
    assert not (target / "not-a-processor.txt").exists()


def test_explicit_checkpoint_evaluates_materialized_bf16_control() -> None:
    config = _pilot() | {
        "quantization_scope": "none",
        "checkpoint": "outputs/bf16-control/checkpoints/000001/pretrained_model",
    }
    command = build_evaluate_command(config)
    assert "--policy.path=outputs/bf16-control/checkpoints/000001/pretrained_model" in command


def test_smoke_checkpoint_can_be_used_for_its_ten_episode_environment_check() -> None:
    config = _pilot() | {
        "episodes": 10,
        "checkpoint": "outputs/cluster-smoke/checkpoints/000001/pretrained_model",
    }
    command = build_evaluate_command(config)
    assert "--eval.n_episodes=1" in command
    assert "--policy.path=outputs/cluster-smoke/checkpoints/000001/pretrained_model" in command


def test_native_bitvla_benchmark_uses_reproducible_wrapper(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config = {
        "architecture": "bitvla",
        "_config_path": str(config_path),
    }
    command = build_benchmark_command(config)
    assert command[1:3] == ("-m", "lerobot_policy_bitwam.bitvla_benchmark")
    assert command[-1] == str(config_path)


def test_screen_uses_ten_episodes_in_a_separate_output_directory(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, output_dir, stage):
        captured.update(command=command, output_dir=output_dir, stage=stage)
        return 0

    config_path = tmp_path / "screen.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_id: pilot",
                "seed: 0",
                "source_checkpoint: lerobot/VLA-JEPA-LIBERO",
                "output_dir: outputs/pilot",
                "quantization_scope: qwen",
                "episodes: 50",
                "task_count: 10",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("lerobot_policy_bitwam.workflows._run_external", fake_run)

    assert run_command("screen", config_path) == 0
    assert captured["stage"] == "screen"
    assert captured["output_dir"] == Path("outputs/pilot/screen-1000")
    assert "--eval.n_episodes=1" in captured["command"]
    assert "--output_dir=outputs/pilot/screen-1000/evaluation" in captured["command"]
    assert "--policy.path=outputs/pilot/checkpoints/last/pretrained_model" in captured["command"]
