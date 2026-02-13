"""Smoke tests for rollout evaluation helpers."""

import importlib.util
from pathlib import Path


def test_rollout_script_imports():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "rollout_eval.py"
    spec = importlib.util.spec_from_file_location("rollout_eval", script_path)
    assert spec is not None and spec.loader is not None


def test_config_files_exist():
    root = Path(__file__).resolve().parents[1]
    expected = [
        root / "configs" / "pusht_quick.yaml",
        root / "configs" / "pusht_standard.yaml",
        root / "configs" / "aloha_sim_quick.yaml",
        root / "configs" / "aloha_sim_standard.yaml",
    ]
    for path in expected:
        assert path.exists(), f"Missing config: {path}"
