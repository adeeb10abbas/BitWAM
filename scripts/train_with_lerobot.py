#!/usr/bin/env python3
"""Canonical pip-first training pipeline for VLABitNet on LeRobot simulation tasks."""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml

# Force local source imports so the current repo is used even if another
# editable install exists elsewhere on the machine.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bit_vla import VLABitNet, VLABitNetConfig
from bit_vla.training import BitNetOptimizer
from bit_vla.utils import SimpleTokenizer, build_task_prompt

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
except ImportError as exc:
    raise ImportError(
        "LeRobot is required. Install with: pip install lerobot"
    ) from exc


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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. Validate your NVIDIA setup with `nvidia-smi`."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def configure_runtime_for_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log"),
        ],
    )


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
        image_key = next(
            (k for k in batch.keys() if "image" in k and isinstance(batch[k], torch.Tensor)),
            None,
        )
    if image_key is None:
        raise KeyError("No image key found in batch")

    images = batch[image_key]
    if images.dim() == 5:
        images = images[:, -1]

    if preset.state_key in batch:
        states = batch[preset.state_key]
    else:
        states = torch.zeros(images.shape[0], preset.state_dim)

    if states.dim() > 2:
        states = states[:, -1]
    states = states.flatten(1)

    actions = batch["action"]
    if actions.dim() > 2:
        actions = actions[:, -1]

    prompt = build_task_prompt(preset.dataset_name)
    text_batch = tokenizer.batch_encode([prompt] * images.shape[0], max_length=max_text_len)

    return {
        "images": images.to(device),
        "states": states.to(device),
        "actions": actions.to(device),
        "token_ids": text_batch["token_ids"].to(device),
        "attention_mask": text_batch["attention_mask"].to(device),
    }


def run_epoch(
    model: VLABitNet,
    loader: DataLoader,
    preset: TaskPreset,
    tokenizer: SimpleTokenizer,
    max_text_len: int,
    device: torch.device,
    optimizer: Optional[BitNetOptimizer] = None,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses: List[float] = []

    for step, raw_batch in enumerate(tqdm(loader, leave=False)):
        batch = to_vla_batch(raw_batch, preset, tokenizer, max_text_len, device)
        pred = model(
            images=batch["images"],
            token_ids=batch["token_ids"],
            attention_mask=batch["attention_mask"],
            states=batch["states"],
        )
        loss = F.smooth_l1_loss(pred, batch["actions"])

        if optimizer is not None:
            optimizer.step_lr_schedule(step)
            optimizer.step(loss)
        losses.append(loss.item())

    return sum(losses) / max(1, len(losses))


def evaluate(
    model: VLABitNet,
    loader: DataLoader,
    preset: TaskPreset,
    tokenizer: SimpleTokenizer,
    max_text_len: int,
    device: torch.device,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = to_vla_batch(raw_batch, preset, tokenizer, max_text_len, device)
            pred = model(
                images=batch["images"],
                token_ids=batch["token_ids"],
                attention_mask=batch["attention_mask"],
                states=batch["states"],
            )
            losses.append(F.smooth_l1_loss(pred, batch["actions"]).item())
    return sum(losses) / max(1, len(losses))


def train(args: argparse.Namespace) -> Dict[str, Any]:
    preset = TASK_PRESETS[args.dataset]
    device = resolve_device(args.device)
    configure_runtime_for_device(device)

    metadata = LeRobotDatasetMetadata(preset.dataset_name)
    fps = float(getattr(metadata, "fps", 10.0))
    dataset = LeRobotDataset(
        preset.dataset_name,
        delta_timestamps=build_delta_timestamps(preset, fps),
        video_backend="pyav",
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    workers = args.num_workers
    pin_memory = device.type == "cuda"
    persistent_workers = workers > 0
    prefetch_factor = args.prefetch_factor if workers > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=default_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=default_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    model_cfg = VLABitNetConfig(
        hidden_dim=args.hidden_dim,
        action_dim=preset.action_dim,
        state_dim=preset.state_dim,
        max_seq_len=args.max_text_len,
        use_tanh_actions=args.use_tanh_actions,
    )
    model = VLABitNet(**asdict(model_cfg)).to(device)
    tokenizer = SimpleTokenizer(vocab_size=model_cfg.vocab_size)

    optimizer = BitNetOptimizer(
        model,
        stage1_lr=args.lr,
        stage2_lr=args.lr * 0.1,
        stage1_steps=max(1, args.epochs * len(train_loader) // 2),
        warmup_steps=min(100, max(10, len(train_loader))),
    )

    best_val = float("inf")
    history: List[Dict[str, float]] = []
    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            preset,
            tokenizer,
            args.max_text_len,
            device,
            optimizer,
        )
        val_loss = evaluate(
            model,
            val_loader,
            preset,
            tokenizer,
            args.max_text_len,
            device,
        )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logging.info("epoch=%d train=%.4f val=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(model_cfg),
                    "dataset": preset.dataset_name,
                    "history": history,
                },
                args.output_dir / "best_model.pt",
            )

    result = {
        "dataset": preset.dataset_name,
        "device": str(device),
        "best_val_loss": best_val,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "profile": args.profile,
        "model_config": asdict(model_cfg),
        "quantization_summary": model.get_quantization_summary(),
        "history": history,
    }
    with open(args.output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if device.type == "cuda":
        result["gpu_peak_memory_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        logging.info("gpu_peak_memory_gb=%.3f", result["gpu_peak_memory_gb"])
    return result


def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must define a mapping: {path}")
    return data


def _merge_args_with_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = _load_yaml(args.config)

    for key in [
        "dataset",
        "profile",
        "epochs",
        "batch_size",
        "hidden_dim",
        "max_text_len",
        "lr",
        "use_tanh_actions",
        "device",
        "num_workers",
        "prefetch_factor",
        "output_dir",
        "run_name",
    ]:
        if getattr(args, key) is None and key in cfg:
            setattr(args, key, cfg[key])

    if args.dataset is None:
        args.dataset = "lerobot/pusht"
    if args.profile is None:
        args.profile = "quick"

    if args.profile == "quick":
        if args.epochs is None:
            args.epochs = 2
        if args.batch_size is None:
            args.batch_size = 16
    else:
        if args.epochs is None:
            args.epochs = 12
        if args.batch_size is None:
            args.batch_size = 32

    if args.hidden_dim is None:
        args.hidden_dim = 256
    if args.max_text_len is None:
        args.max_text_len = 32
    if args.lr is None:
        args.lr = 3e-4
    if args.use_tanh_actions is None:
        args.use_tanh_actions = False
    if args.device is None:
        args.device = "auto"
    if args.num_workers is None:
        args.num_workers = 4 if torch.cuda.is_available() else 0
    if args.prefetch_factor is None:
        args.prefetch_factor = 2

    base_output = Path(args.output_dir) if args.output_dir is not None else Path("outputs") / "vla_pipeline"
    run_name = args.run_name if args.run_name is not None else f"{args.dataset.replace('/', '_')}_{args.profile}"
    args.output_dir = base_output / run_name
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VLABitNet on LeRobot sim tasks")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--dataset", type=str, default=None, choices=list(TASK_PRESETS.keys()))
    parser.add_argument("--profile", type=str, default=None, choices=["quick", "standard"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--max_text_len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--use_tanh_actions", action="store_true", default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()
    return _merge_args_with_config(args)


def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir)
    logging.info("Starting run with args=%s", vars(args))
    results = train(args)
    logging.info("Training finished. best_val_loss=%.4f", results["best_val_loss"])


if __name__ == "__main__":
    main()
