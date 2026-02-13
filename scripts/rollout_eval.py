#!/usr/bin/env python3
"""Open-loop rollout-style evaluation for trained VLABitNet checkpoints."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

# Force local source imports so the current repo is used even if another
# editable install exists elsewhere on the machine.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bit_vla import VLABitNet
from bit_vla.utils import SimpleTokenizer, build_task_prompt

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
except ImportError as exc:
    raise ImportError("LeRobot is required. Install with: pip install lerobot") from exc


@dataclass
class TaskPreset:
    dataset_name: str
    action_dim: int
    state_dim: int
    chunk_size: int
    image_keys: List[str]
    state_key: str = "observation.state"


TASK_PRESETS: Dict[str, TaskPreset] = {
    "lerobot/pusht": TaskPreset(
        dataset_name="lerobot/pusht",
        action_dim=2,
        state_dim=2,
        chunk_size=16,
        image_keys=["observation.image"],
    ),
    "lerobot/aloha_sim_insertion_human": TaskPreset(
        dataset_name="lerobot/aloha_sim_insertion_human",
        action_dim=14,
        state_dim=14,
        chunk_size=50,
        image_keys=[
            "observation.images.cam_high",
            "observation.images.top",
            "observation.image",
        ],
    ),
}


def build_delta_timestamps(preset: TaskPreset, fps: float) -> Dict[str, List[float]]:
    delta = {
        preset.state_key: [0.0],
        "action": [i / max(1.0, fps) for i in range(preset.chunk_size)],
    }
    for k in preset.image_keys:
        delta[k] = [0.0]
    return delta


def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out


def find_first_key(batch: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in batch:
            return key
    return None


def to_vla_batch(
    batch: Dict[str, Any],
    preset: TaskPreset,
    tokenizer: SimpleTokenizer,
    max_text_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    image_key = find_first_key(batch, preset.image_keys)
    if image_key is None:
        image_key = next((k for k in batch if "image" in k and isinstance(batch[k], torch.Tensor)), None)
    if image_key is None:
        raise KeyError("No image key found in batch")

    images = batch[image_key]
    if images.dim() == 5:
        images = images[:, -1]

    states = batch.get(preset.state_key, torch.zeros(images.shape[0], preset.state_dim))
    if states.dim() > 2:
        states = states[:, -1]
    states = states.flatten(1)

    actions = batch["action"]
    if actions.dim() > 2:
        actions = actions[:, -1]

    prompt = build_task_prompt(preset.dataset_name)
    tokens = tokenizer.batch_encode([prompt] * images.shape[0], max_length=max_text_len)
    return {
        "images": images.to(device),
        "states": states.to(device),
        "actions": actions.to(device),
        "token_ids": tokens["token_ids"].to(device),
        "attention_mask": tokens["attention_mask"].to(device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollout-style checkpoint evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None, choices=list(TASK_PRESETS.keys()))
    parser.add_argument("--max_text_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--pass_l1", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if args.dataset is None and "dataset" in cfg:
            args.dataset = cfg["dataset"]
        if "max_text_len" in cfg and parser.get_default("max_text_len") == args.max_text_len:
            args.max_text_len = int(cfg["max_text_len"])
        if "batch_size" in cfg and parser.get_default("batch_size") == args.batch_size:
            args.batch_size = int(cfg["batch_size"])

    return args


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    dataset_name = args.dataset or ckpt.get("dataset")
    if dataset_name not in TASK_PRESETS:
        raise ValueError(f"Dataset not recognized: {dataset_name}")
    preset = TASK_PRESETS[dataset_name]

    cfg = ckpt.get("config", {})
    model = VLABitNet(**cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokenizer = SimpleTokenizer(vocab_size=cfg.get("vocab_size", 8192))
    metadata = LeRobotDatasetMetadata(preset.dataset_name)
    fps = float(getattr(metadata, "fps", 10.0))
    dataset = LeRobotDataset(
        preset.dataset_name,
        delta_timestamps=build_delta_timestamps(preset, fps),
        video_backend="pyav",
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=default_collate,
    )

    mse_vals: List[float] = []
    l1_vals: List[float] = []
    tol_hits = 0
    tol_total = 0

    with torch.no_grad():
        for batch_idx, raw in enumerate(loader):
            if batch_idx >= args.num_batches:
                break
            batch = to_vla_batch(raw, preset, tokenizer, args.max_text_len, device)
            pred = model(
                images=batch["images"],
                token_ids=batch["token_ids"],
                attention_mask=batch["attention_mask"],
                states=batch["states"],
            )

            err = pred - batch["actions"]
            mse_vals.append(F.mse_loss(pred, batch["actions"]).item())
            l1_vals.append(F.l1_loss(pred, batch["actions"]).item())

            hit = (err.abs() < args.pass_l1).float()
            tol_hits += int(hit.sum().item())
            tol_total += int(hit.numel())

    mean_mse = sum(mse_vals) / max(1, len(mse_vals))
    mean_l1 = sum(l1_vals) / max(1, len(l1_vals))
    within_tol_pct = 100.0 * tol_hits / max(1, tol_total)

    status = "PASS" if mean_l1 <= args.pass_l1 else "WARN"
    report = {
        "status": status,
        "dataset": preset.dataset_name,
        "checkpoint": str(args.checkpoint),
        "num_batches": len(mse_vals),
        "mean_mse": mean_mse,
        "mean_l1": mean_l1,
        "pass_l1_threshold": args.pass_l1,
        "within_threshold_percentage": within_tol_pct,
    }

    out_path = args.output
    if out_path is None:
        out_path = args.checkpoint.parent / "rollout_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
