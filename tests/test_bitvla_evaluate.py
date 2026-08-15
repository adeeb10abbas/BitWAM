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
