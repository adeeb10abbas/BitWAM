from pathlib import Path

import pytest

from lerobot_policy_bitwam.bitvla_benchmark import deployment_artifacts, summarize_latencies


def test_latency_summary_interpolates_percentiles() -> None:
    summary = summarize_latencies([1, 2, 3, 4, 5])
    assert summary["p50_ms"] == 3
    assert summary["p95_ms"] == pytest.approx(4.8)
    assert summary["mean_ms"] == 3


def test_deployment_artifacts_exclude_training_only_files(tmp_path: Path) -> None:
    for name, contents in {
        "config.json": "config",
        "model-00001-of-00001.safetensors": "weights",
        "action_head--1_checkpoint.pt": "action",
        "proprio_projector--1_checkpoint.pt": "proprio",
        "world_model--1_checkpoint.pt": "world",
        "optimizer--1_checkpoint.pt": "optimizer",
        "config.json.back.123": "backup",
    }.items():
        (tmp_path / name).write_text(contents, encoding="utf-8")

    names = {item["name"] for item in deployment_artifacts(tmp_path)}
    assert names == {
        "config.json",
        "model-00001-of-00001.safetensors",
        "action_head--1_checkpoint.pt",
        "proprio_projector--1_checkpoint.pt",
    }
