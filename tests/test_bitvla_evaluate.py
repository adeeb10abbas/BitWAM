from pathlib import Path

import pytest

from lerobot_policy_bitwam.bitvla_evaluate import build_upstream_eval_argv
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
