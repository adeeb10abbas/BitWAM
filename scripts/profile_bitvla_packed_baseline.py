#!/usr/bin/env python3
"""Profile the unmodified BitVLA action request before packed-kernel work.

Run this from BitVLA's Python 3.10 environment on an idle GPU pod.  The script
does not modify a checkpoint: BitVLA's two helper functions that otherwise sync
source files into a local checkpoint are explicitly disabled.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="put the black bowl on the plate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream_root = args.upstream_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep generated caches out of the checkout and make the upstream modules importable.
    os.environ.setdefault("HF_HOME", "/tmp/bitwam-profile-hf")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/bitwam-inductor")
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/bitwam-triton")
    default_cpath = (
        "/data/users/ali/.local/python-dev-3.10/usr/include:"
        "/data/users/ali/.local/python-dev-3.10/usr/include/python3.10"
    )
    os.environ.setdefault("CPATH", default_cpath)
    for path in (
        upstream_root / "openvla-oft" / "bitvla",
        upstream_root / "openvla-oft",
        upstream_root / "transformers" / "src",
    ):
        sys.path.insert(0, str(path))

    from experiments.robot import bitnet_utils, openvla_utils

    # These upstream helpers make backups and rewrite shared local checkpoints.
    bitnet_utils.update_auto_map = lambda *args, **kwargs: None
    bitnet_utils.check_model_logic_mismatch = lambda *args, **kwargs: None

    cfg = SimpleNamespace(
        pretrained_checkpoint=str(checkpoint),
        load_in_8bit=False,
        load_in_4bit=False,
        task_suite_name="libero_10",
        unnorm_key="libero_10_no_noops",
        num_images_in_input=2,
        use_proprio=True,
        num_open_loop_steps=8,
        use_l1_regression=True,
        use_diffusion=False,
        num_diffusion_steps=50,
        use_film=False,
        center_crop=True,
    )
    model = bitnet_utils.get_bitnet_vla(cfg)
    model.set_constant(
        image_token_idx=128010,
        proprio_pad_idx=128011,
        ignore_idx=-100,
        action_token_begin_idx=128011,
        stop_index=128001,
    )
    action_head = openvla_utils.get_action_head(cfg, model.config.text_config.hidden_size)
    proprio_projector = openvla_utils.get_proprio_projector(
        cfg, model.config.text_config.hidden_size, proprio_dim=8
    )
    processor = openvla_utils.get_processor(cfg)
    observation = {
        "full_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
    }

    def query() -> object:
        return bitnet_utils.get_bitnet_vla_action(
            cfg,
            model,
            processor,
            observation,
            args.task,
            action_head=action_head,
            proprio_projector=proprio_projector,
        )

    for _ in range(2):
        query()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profile:
        query()
        torch.cuda.synchronize()

    trace_path = output_dir / "dense-native-626-prefill.trace.json"
    table_path = output_dir / "dense-native-626-prefill.profiler-table.txt"
    profile.export_chrome_trace(str(trace_path))
    table_path.write_text(
        "dense native BitVLA 626-token prefill / one timed query\n"
        + profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=45)
        + "\n--- CPU ---\n"
        + profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
    )
    print(f"trace={trace_path}")
    print(f"table={table_path}")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
