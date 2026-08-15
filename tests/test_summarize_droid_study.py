import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    script = Path(__file__).parents[1] / "scripts/summarize_droid_study.py"
    spec = importlib.util.spec_from_file_location("summarize_droid_study", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(run_root: Path, relative: str, *rows: dict) -> None:
    path = run_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_summary_applies_stage_p_and_m_gates(tmp_path: Path) -> None:
    module = _module()
    base = {"micro_step": 100, "action_loss": 0.2, "world_action_conditioning_gap": 0.0}
    _write(
        tmp_path,
        module.RUNS["holdout_initialization"],
        base | {"world_cosine_similarity": 0.1},
    )
    _write(
        tmp_path,
        module.RUNS["holdout_pretrain_normal"],
        base | {"world_cosine_similarity": 0.8, "world_action_conditioning_gap": 0.2},
    )
    _write(
        tmp_path,
        module.RUNS["holdout_pretrain_shuffled_input"],
        base | {"world_cosine_similarity": 0.6},
    )
    _write(
        tmp_path,
        module.RUNS["midtrain"],
        base | {"micro_step": 10, "world_cosine_similarity": 0.7},
        base
        | {
            "micro_step": 200,
            "action_loss": 0.205,
            "world_cosine_similarity": 0.8,
            "world_action_conditioning_gap": 0.1,
        },
    )

    result = module.summarize(tmp_path)
    assert result["gates"]["stage_p"]["status"] == "passed"
    assert result["gates"]["stage_m"]["status"] == "passed"


def test_summary_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    module = _module()
    _write(tmp_path, module.RUNS["pretrain"], {"world_loss": float("nan")})
    with pytest.raises(ValueError, match="non-finite"):
        module.summarize(tmp_path)
