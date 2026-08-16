from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot_policy_bitwam.bitvla_evaluate import _enable_packed_runtime, build_upstream_eval_argv
from lerobot_policy_bitwam.workflows import build_evaluate_command


def _config(tmp_path: Path) -> dict:
    return {
        "architecture": "bitvla",
        "checkpoint": "/models/bitwam",
        "output_dir": "/runs/bitwam-eval",
        "run_id": "bitwam-eval",
        "episodes": 50,
        "task_count": 10,
        "seed": 11,
        "_config_path": str(tmp_path / "eval.yaml"),
    }


def test_build_upstream_eval_argv_uses_official_bitnet_harness(tmp_path: Path) -> None:
    argv = build_upstream_eval_argv(_config(tmp_path))
    assert argv[argv.index("--pretrained_checkpoint") + 1] == "/models/bitwam"
    assert argv[argv.index("--num_trials_per_task") + 1] == "5"
    assert argv[argv.index("--model_family") + 1] == "bitnet"
    assert argv[argv.index("--use_wandb") + 1] == "False"
    assert argv[argv.index("--seed") + 1] == "11"


def test_build_upstream_eval_argv_requires_even_task_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        build_upstream_eval_argv(_config(tmp_path) | {"episodes": 11})


def test_workflow_routes_bitvla_evaluation_to_native_wrapper(tmp_path: Path) -> None:
    command = build_evaluate_command(_config(tmp_path))
    assert command[1:3] == ("-m", "lerobot_policy_bitwam.bitvla_evaluate")
    assert command[-1] == str(tmp_path / "eval.yaml")


def test_packed_runtime_wraps_upstream_model_initialization(monkeypatch, capsys) -> None:
    report = Mock()
    report.to_dict.return_value = {"packed_layers": 2}
    pack = Mock(return_value=report)
    monkeypatch.setattr("lerobot_policy_bitwam.bitvla_packing.pack_bitlinear_weights", pack)
    evaluator = SimpleNamespace(initialize_model=lambda _cfg: ("model", "head"))

    _enable_packed_runtime(
        evaluator,
        {"packed_scope": "text", "packed_runtime_backend": "eager_unpack"},
    )

    assert evaluator.initialize_model("cfg") == ("model", "head")
    pack.assert_called_once_with("model", scope="text")
    assert '"backend": "eager_unpack"' in capsys.readouterr().out


def test_packed_runtime_routes_direct_triton_candidate(monkeypatch, capsys) -> None:
    report = Mock()
    report.to_dict.return_value = {"packed_layers": 2}
    direct = Mock(return_value=2)
    monkeypatch.setattr(
        "lerobot_policy_bitwam.bitvla_packing.pack_bitlinear_weights",
        Mock(return_value=report),
    )
    monkeypatch.setattr(
        "lerobot_policy_bitwam.bitvla_packing.enable_triton_direct_bitlinear_runtime",
        direct,
    )
    evaluator = SimpleNamespace(initialize_model=lambda _cfg: ("model", "head"))

    _enable_packed_runtime(
        evaluator,
        {
            "packed_scope": "text",
            "packed_runtime_backend": "triton_direct_bf16",
            "packed_activation_backend": "hybrid",
        },
    )

    assert evaluator.initialize_model("cfg") == ("model", "head")
    direct.assert_called_once_with(
        "model",
        activation_backend="hybrid",
        bf16_candidate=True,
    )
    assert '"backend": "triton_direct_bf16"' in capsys.readouterr().out


def test_packed_runtime_routes_production_w2a8_backend(monkeypatch, capsys) -> None:
    report = Mock()
    report.to_dict.return_value = {"packed_layers": 2}
    direct = Mock(return_value=2)
    monkeypatch.setattr(
        "lerobot_policy_bitwam.bitvla_packing.pack_bitlinear_weights",
        Mock(return_value=report),
    )
    monkeypatch.setattr(
        "lerobot_policy_bitwam.bitvla_packing.enable_direct_w2a8_bitlinear_runtime",
        direct,
    )
    evaluator = SimpleNamespace(initialize_model=lambda _cfg: ("model", "head"))

    _enable_packed_runtime(
        evaluator,
        {
            "packed_scope": "all",
            "packed_runtime_backend": "triton_w2a8",
            "packed_activation_backend": "hybrid",
            "packed_small_batch_threshold": 4,
        },
    )

    assert evaluator.initialize_model("cfg") == ("model", "head")
    direct.assert_called_once_with(
        "model",
        activation_backend="hybrid",
        small_batch_threshold=4,
    )
    output = capsys.readouterr().out
    assert '"backend": "triton_w2a8"' in output
    assert '"packed_layout": "dp4a"' in output


def test_packed_runtime_loads_dp4a_artifact_without_dense_model(monkeypatch, tmp_path) -> None:
    artifact = tmp_path / "packed"
    artifact.mkdir()
    (artifact / "tensors.pt").write_bytes(b"packed")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "dataset_statistics.json").write_text(
        '{"libero_10_no_noops": {"action": {}}}',
        encoding="utf-8",
    )
    model = SimpleNamespace(norm_stats=None)
    moved_model = SimpleNamespace(norm_stats=None)
    manifest = SimpleNamespace(
        packing={"packed_layout": "dp4a", "packed_layers": 2},
        packed_layers=({"module": "one"}, {"module": "two"}),
    )
    build = Mock(return_value=model)
    load = Mock(return_value=manifest)
    move = Mock(return_value=moved_model)
    direct = Mock(return_value=2)
    pack = Mock()
    monkeypatch.setattr(
        "lerobot_policy_bitwam.bitvla_packed_checkpoint.build_bitvla_topology",
        build,
    )
    monkeypatch.setattr(
        "lerobot_policy_bitwam.bitvla_packed_checkpoint.load_packed_checkpoint",
        load,
    )
    monkeypatch.setattr(
        "lerobot_policy_bitwam.bitvla_packed_checkpoint.move_packed_bitvla_to_device",
        move,
    )
    monkeypatch.setattr(
        "lerobot_policy_bitwam.bitvla_packing.enable_direct_w2a8_bitlinear_runtime",
        direct,
    )
    monkeypatch.setattr(
        "lerobot_policy_bitwam.bitvla_packing.pack_bitlinear_weights",
        pack,
    )
    evaluator = SimpleNamespace(get_model=Mock(side_effect=AssertionError("dense load")))
    evaluator.initialize_model = lambda cfg: (evaluator.get_model(cfg), "head")

    _enable_packed_runtime(
        evaluator,
        {
            "checkpoint": str(checkpoint),
            "packed_artifact": str(artifact),
            "packed_runtime_backend": "triton_w2a8",
            "packed_activation_backend": "triton",
        },
    )

    assert evaluator.initialize_model("cfg") == (moved_model, "head")
    build.assert_called_once_with(checkpoint.resolve())
    load.assert_called_once_with(
        model,
        artifact.resolve(),
        expected_source_metadata={"checkpoint": str(checkpoint.resolve())},
    )
    move.assert_called_once_with(model, "cuda:0")
    direct.assert_called_once_with(
        moved_model,
        activation_backend="triton",
        small_batch_threshold=8,
    )
    pack.assert_not_called()
    assert model.norm_stats == {"libero_10_no_noops": {"action": {}}}
    assert evaluator._bitwam_packing_report["direct_artifact_load"] is True
