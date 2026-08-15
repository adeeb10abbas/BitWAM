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
        "world_head_precision": "ternary",
        "world_contrastive_weight": 1.0,
        "world_contrastive_margin": 0.05,
        "action_checkpoint": "/checkpoints/action.pt",
        "proprio_checkpoint": "/checkpoints/proprio.pt",
        "rlds_split": "train[:99%]",
        "dataset_statistics_path": "/data/statistics.json",
        "world_action_mode": "shuffled",
        "freeze_policy": True,
        "seed": 11,
        "vla_path": "/models/control",
        "use_proprio": True,
        "wandb_log_freq": 7,
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
    assert runtime.world_head_precision == "ternary"
    assert runtime.world_contrastive_weight == 1.0
    assert runtime.world_contrastive_margin == 0.05
    assert runtime.action_checkpoint == Path("/checkpoints/action.pt")
    assert runtime.proprio_checkpoint == Path("/checkpoints/proprio.pt")
    assert runtime.rlds_split == "train[:99%]"
    assert runtime.dataset_statistics_path == Path("/data/statistics.json")
    assert runtime.world_action_mode == "shuffled"
    assert runtime.metrics_log_frequency == 7
    assert runtime.seed == 11


def test_build_upstream_argv_excludes_bitwam_fields(tmp_path: Path) -> None:
    argv = build_upstream_argv(_config(tmp_path))
    assert argv[0] == "finetune_bitnet.py"
    assert argv[argv.index("--vla_path") + 1] == "/models/control"
    assert argv[argv.index("--use_proprio") + 1] == "True"
    assert "--world_loss_weight" not in argv
    assert "--world_head_precision" not in argv
    assert "--world_contrastive_weight" not in argv
    assert "--world_contrastive_margin" not in argv
    assert "--action_checkpoint" not in argv
    assert "--proprio_checkpoint" not in argv
    assert "--rlds_split" not in argv
    assert "--dataset_statistics_path" not in argv
    assert "--world_action_mode" not in argv
    assert "--num_processes" not in argv


def test_runtime_config_allows_zero_weight_for_action_only_ablation(tmp_path: Path) -> None:
    runtime = parse_runtime_config(_config(tmp_path) | {"world_loss_weight": 0})
    assert runtime.world_loss_weight == 0


def test_runtime_config_rejects_negative_world_loss(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        parse_runtime_config(_config(tmp_path) | {"world_loss_weight": -0.1})


def test_runtime_config_rejects_unknown_world_head_precision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="world_head_precision"):
        parse_runtime_config(_config(tmp_path) | {"world_head_precision": "int4"})


def test_runtime_config_rejects_unknown_world_action_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="world_action_mode"):
        parse_runtime_config(_config(tmp_path) | {"world_action_mode": "permuted"})


def test_runtime_config_rejects_nonpositive_metrics_frequency(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="wandb_log_freq"):
        parse_runtime_config(_config(tmp_path) | {"wandb_log_freq": 0})


@pytest.mark.parametrize("split", ("val", "train+test", "train[:99%] ", "train[::2]"))
def test_runtime_config_rejects_non_deterministic_train_slice(tmp_path: Path, split: str) -> None:
    with pytest.raises(ValueError, match="rlds_split"):
        parse_runtime_config(_config(tmp_path) | {"rlds_split": split})


@pytest.mark.parametrize(
    ("field", "value"),
    (("world_contrastive_weight", -1.0), ("world_contrastive_margin", -0.1)),
)
def test_runtime_config_rejects_negative_contrastive_settings(
    tmp_path: Path, field: str, value: float
) -> None:
    with pytest.raises(ValueError, match="contrastive"):
        parse_runtime_config(_config(tmp_path) | {field: value})
