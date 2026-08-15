from pathlib import Path

import pytest

from lerobot_policy_bitwam.bitvla_train import (
    build_upstream_argv,
    parse_runtime_config,
)

UPSTREAM_REVISION = "8afac0260b3748b14657a69ec58e3d9f0d6da3a7"


def _config(tmp_path: Path) -> dict:
    script = tmp_path / "openvla-oft/vla-scripts/finetune_bitnet.py"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    return {
        "architecture": "bitvla",
        "upstream_root": str(tmp_path),
        "upstream_revision": UPSTREAM_REVISION,
        "world_loss_weight": 0.25,
        "world_learning_rate": 2e-4,
        "freeze_policy": True,
        "seed": 11,
        "vla_path": "/models/control",
        "use_proprio": True,
        "max_steps": 102000,
        "_config_path": "/configs/stage1.yaml",
        "output_dir": "/runs/stage1",
        "num_processes": 4,
    }


def test_parse_runtime_config_keeps_world_stage_separate(tmp_path: Path) -> None:
    runtime = parse_runtime_config(_config(tmp_path))
    assert runtime.upstream_root == tmp_path.resolve()
    assert runtime.freeze_policy
    assert runtime.world_loss_weight == 0.25
    assert runtime.world_learning_rate == 2e-4
    assert runtime.seed == 11


def test_build_upstream_argv_excludes_bitwam_fields(tmp_path: Path) -> None:
    argv = build_upstream_argv(_config(tmp_path))
    assert argv[0] == "finetune_bitnet.py"
    assert argv[argv.index("--vla_path") + 1] == "/models/control"
    assert argv[argv.index("--use_proprio") + 1] == "True"
    assert "--world_loss_weight" not in argv
    assert "--num_processes" not in argv


def test_runtime_config_requires_positive_world_loss(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        parse_runtime_config(_config(tmp_path) | {"world_loss_weight": 0})
