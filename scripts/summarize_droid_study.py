#!/usr/bin/env python3
"""Compile DROID training and holdout JSONL into one gate-aware summary."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

RUNS = {
    "pretrain": "bitwam-droid-pretrain/metrics.jsonl",
    "zero_action_pretrain": "bitwam-droid-pretrain-zero-action/metrics.jsonl",
    "shuffled_action_pretrain": "bitwam-droid-pretrain-shuffled-action/metrics.jsonl",
    "holdout_initialization": "bitwam-droid-holdout-initialization/metrics.jsonl",
    "holdout_pretrain_normal": "bitwam-droid-holdout-pretrain-normal/metrics.jsonl",
    "holdout_pretrain_zero_input": "bitwam-droid-holdout-pretrain-zero/metrics.jsonl",
    "holdout_pretrain_shuffled_input": "bitwam-droid-holdout-pretrain-shuffled/metrics.jsonl",
    "holdout_zero_pretrain": "bitwam-droid-holdout-zero-pretrain/metrics.jsonl",
    "holdout_shuffled_pretrain": "bitwam-droid-holdout-shuffled-pretrain/metrics.jsonl",
    "midtrain": "bitwam-droid-midtrain/metrics.jsonl",
    "action_only_midtrain": "bitvla-action-only-droid-midtrain/metrics.jsonl",
    "posttrain": "bitwam-droid-libero-posttrain/metrics.jsonl",
    "action_only_posttrain": "bitvla-action-only-droid-libero-posttrain/metrics.jsonl",
    "no_mid_posttrain": "bitwam-droid-pretrain-libero-posttrain/metrics.jsonl",
}

SYSTEM_ENDPOINT_FIELDS = (
    "micro_step",
    "elapsed_forward_seconds",
    "global_examples_seen",
    "global_examples_per_second",
)
SYSTEM_PEAK_FIELDS = (
    "cuda_max_memory_allocated_bytes",
    "cuda_max_memory_reserved_bytes",
)


def _read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"metrics file is empty: {path}")
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise ValueError(f"non-finite metric {key} in {path}")
    return rows


def _summarize_systems(rows: list[dict]) -> dict:
    """Extract directly comparable run-local throughput and allocator evidence."""
    summary = {
        key: rows[-1][key] for key in SYSTEM_ENDPOINT_FIELDS if key in rows[-1]
    }
    for key in SYSTEM_PEAK_FIELDS:
        values = [row[key] for row in rows if key in row]
        if values:
            summary[key] = max(values)
    return summary


def summarize(run_root: Path) -> dict:
    """Return available endpoints plus preregistered promotion-gate decisions."""
    runs = {}
    all_rows = {}
    for name, relative in RUNS.items():
        path = run_root / relative
        if path.is_file():
            rows = _read_rows(path)
            all_rows[name] = rows
            runs[name] = {
                "path": str(path),
                "rows": len(rows),
                "first": rows[0],
                "last": rows[-1],
                "systems": _summarize_systems(rows),
            }

    gates: dict[str, dict] = {
        "stage_p": {"status": "pending"},
        "stage_m": {"status": "pending"},
    }
    required_p = {
        "holdout_initialization",
        "holdout_pretrain_normal",
        "holdout_pretrain_shuffled_input",
    }
    if required_p <= all_rows.keys():
        initial = all_rows["holdout_initialization"][-1]
        normal = all_rows["holdout_pretrain_normal"][-1]
        shuffled = all_rows["holdout_pretrain_shuffled_input"][-1]
        conditions = {
            "cosine_above_initialization": (
                normal["world_cosine_similarity"] > initial["world_cosine_similarity"]
            ),
            "cosine_above_shuffled_input": (
                normal["world_cosine_similarity"] > shuffled["world_cosine_similarity"]
            ),
            "positive_action_conditioning_gap": (
                normal["world_action_conditioning_gap"] > 0
            ),
        }
        gates["stage_p"] = {
            "status": "passed" if all(conditions.values()) else "failed",
            "conditions": conditions,
        }

    if "midtrain" in all_rows:
        first = all_rows["midtrain"][0]
        last = all_rows["midtrain"][-1]
        conditions = {
            "positive_action_conditioning_gap": last["world_action_conditioning_gap"] > 0,
            "action_l1_within_5_percent_of_start": (
                last["action_loss"] <= 1.05 * first["action_loss"]
            ),
        }
        gates["stage_m"] = {
            "status": "passed" if all(conditions.values()) else "failed",
            "conditions": conditions,
        }

    return {
        "schema_version": 1,
        "run_root": str(run_root.resolve()),
        "runs": runs,
        "gates": gates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = summarize(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "gates": payload["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
