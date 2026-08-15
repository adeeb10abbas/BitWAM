#!/usr/bin/env python3
"""Precompute BitVLA/OXE normalization statistics for a DROID TFDS slice."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train[:99%]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--num-parallel-reads",
        type=int,
        default=16,
        help="Maximum concurrent RLDS file readers (default: 16).",
    )
    parser.add_argument(
        "--num-parallel-calls",
        type=int,
        default=16,
        help="Maximum concurrent trajectory-map calls (default: 16).",
    )
    return parser


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _bounded_parallelism(requested: int | None, limit: int) -> int:
    """Replace AUTOTUNE and cap explicit tf.data parallelism for full DROID."""
    if requested is None or requested < 1:
        return limit
    return min(requested, limit)


def main() -> int:
    args = build_parser().parse_args()
    if not re.fullmatch(r"train(?:\[\d*%?:\d*%?\])?", args.split):
        raise SystemExit("split must be a deterministic slice of the train split")
    if args.num_parallel_reads < 1 or args.num_parallel_calls < 1:
        raise SystemExit("parallel read and call limits must be positive")
    upstream_root = args.upstream_root.expanduser().resolve()
    openvla_root = upstream_root / "openvla-oft"
    if not (openvla_root / "prismatic/vla/datasets/datasets.py").is_file():
        raise SystemExit(f"Incomplete OpenVLA checkout: {upstream_root}")
    sys.path[:0] = [str(openvla_root / "bitvla"), str(openvla_root), str(upstream_root)]
    os.chdir(openvla_root)
    sys.argv = ["prepare_droid_statistics.py", "droid"]

    import dlimp as dl

    original_from_rlds = dl.DLataset.from_rlds
    original_traj_map = dl.DLataset.traj_map

    def from_rlds(builder, split="train", shuffle=True, num_parallel_reads=-1):
        builder_name = str(getattr(builder, "name", "")).lower()
        if ("droid" in builder_name or builder_name == "r2d2_faceblur") and split in {
            "all",
            "train",
        }:
            split = args.split
        return original_from_rlds(
            builder,
            split=split,
            shuffle=shuffle,
            num_parallel_reads=_bounded_parallelism(
                num_parallel_reads, args.num_parallel_reads
            ),
        )

    dl.DLataset.from_rlds = staticmethod(from_rlds)

    def traj_map(self, fn, num_parallel_calls=-1, **kwargs):
        return original_traj_map(
            self,
            fn,
            num_parallel_calls=_bounded_parallelism(
                num_parallel_calls, args.num_parallel_calls
            ),
            **kwargs,
        )

    dl.DLataset.traj_map = traj_map

    from prismatic.vla.datasets.rlds import dataset as rlds_dataset

    original_statistics = rlds_dataset.get_dataset_statistics

    def get_dataset_statistics(dataset, hash_dependencies, save_dir=None):
        dependencies = tuple(hash_dependencies) + (f"rlds_split={args.split}",)
        return original_statistics(dataset, dependencies, save_dir)

    rlds_dataset.get_dataset_statistics = get_dataset_statistics

    from prismatic.vla.datasets.datasets import RLDSDataset

    started = time.time()
    dataset = RLDSDataset(
        args.data_root,
        "droid",
        lambda batch: batch,
        resize_resolution=(224, 224),
        shuffle_buffer_size=100,
        image_aug=False,
    )
    manifest = {
        "schema_version": 1,
        "dataset": "droid",
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        "dataset_length": len(dataset),
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(dataset.dataset_statistics, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    manifest_path = args.output.with_suffix(f"{args.output.suffix}.manifest.json")
    temporary_manifest = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
