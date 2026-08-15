"""Staged future-latent training on the native ternary BitVLA backbone."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import nn

from lerobot_policy_bitwam.bitvla_world import LatentWorldModelHead
from lerobot_policy_bitwam.workflows import load_config

_RUNTIME_FIELDS = {
    "action_checkpoint",
    "architecture",
    "config_revision",
    "freeze_policy",
    "metrics_path",
    "num_processes",
    "optimizer_checkpoint",
    "proprio_checkpoint",
    "save_backbone",
    "stage",
    "upstream_revision",
    "upstream_root",
    "world_action_embedding_dim",
    "world_checkpoint",
    "world_contrastive_margin",
    "world_contrastive_weight",
    "world_hidden_dim",
    "world_learning_rate",
    "world_loss_weight",
    "world_head_precision",
}


@dataclass(frozen=True)
class BitVLARuntimeConfig:
    """BitWAM-only settings layered over BitVLA's upstream trainer config."""

    upstream_root: Path
    upstream_revision: str
    stage: str
    freeze_policy: bool
    world_loss_weight: float
    world_learning_rate: float
    world_hidden_dim: int
    world_action_embedding_dim: int
    world_head_precision: str
    world_contrastive_weight: float
    world_contrastive_margin: float
    save_backbone: bool
    world_checkpoint: Path | None
    action_checkpoint: Path | None
    proprio_checkpoint: Path | None
    optimizer_checkpoint: Path | None
    metrics_path: Path | None
    seed: int


class _FrozenModule(nn.Module):
    """Expose the ``.module`` interface expected by the upstream DDP trainer."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


_RUNTIME: BitVLARuntimeConfig | None = None
_WORLD_HEAD: nn.Module | None = None
_OPTIMIZER: torch.optim.Optimizer | None = None
_SCHEDULER: Any = None
_METRICS = {
    "world_loss": deque(maxlen=128),
    "world_cosine_similarity": deque(maxlen=128),
    "world_shuffled_action_loss": deque(maxlen=128),
    "world_action_conditioning_gap": deque(maxlen=128),
    "world_contrastive_loss": deque(maxlen=128),
    "action_loss": deque(maxlen=128),
}
_FORWARD_COUNT = 0


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required BitVLA configuration field: {key}")
    return config[key]


def parse_runtime_config(config: dict[str, Any]) -> BitVLARuntimeConfig:
    """Validate and extract fields owned by the BitWAM integration."""
    upstream_root = Path(_required(config, "upstream_root")).expanduser().resolve()
    if not (upstream_root / "openvla-oft/vla-scripts/finetune_bitnet.py").is_file():
        raise ValueError(f"BitVLA upstream checkout is incomplete: {upstream_root}")
    weight = float(config.get("world_loss_weight", 0.1))
    if weight < 0:
        raise ValueError("world_loss_weight must be non-negative for BitVLA WAM training")
    world_checkpoint = config.get("world_checkpoint")
    action_checkpoint = config.get("action_checkpoint")
    proprio_checkpoint = config.get("proprio_checkpoint")
    optimizer_checkpoint = config.get("optimizer_checkpoint")
    metrics_path = config.get("metrics_path")
    world_head_precision = str(config.get("world_head_precision", "bf16"))
    if world_head_precision not in {"bf16", "ternary"}:
        raise ValueError("world_head_precision must be one of: bf16, ternary")
    contrastive_weight = float(config.get("world_contrastive_weight", 0.0))
    contrastive_margin = float(config.get("world_contrastive_margin", 0.05))
    if contrastive_weight < 0 or contrastive_margin < 0:
        raise ValueError("world contrastive weight and margin must be non-negative")
    return BitVLARuntimeConfig(
        upstream_root=upstream_root,
        upstream_revision=str(_required(config, "upstream_revision")),
        stage=str(config.get("stage", "joint_posttrain")),
        freeze_policy=bool(config.get("freeze_policy", False)),
        world_loss_weight=weight,
        world_learning_rate=float(config.get("world_learning_rate", 1e-4)),
        world_hidden_dim=int(config.get("world_hidden_dim", 2048)),
        world_action_embedding_dim=int(config.get("world_action_embedding_dim", 256)),
        world_head_precision=world_head_precision,
        world_contrastive_weight=contrastive_weight,
        world_contrastive_margin=contrastive_margin,
        save_backbone=bool(config.get("save_backbone", True)),
        world_checkpoint=Path(world_checkpoint) if world_checkpoint else None,
        action_checkpoint=Path(action_checkpoint) if action_checkpoint else None,
        proprio_checkpoint=Path(proprio_checkpoint) if proprio_checkpoint else None,
        optimizer_checkpoint=Path(optimizer_checkpoint) if optimizer_checkpoint else None,
        metrics_path=Path(metrics_path) if metrics_path else None,
        seed=int(config.get("seed", 0)),
    )


def build_upstream_argv(config: dict[str, Any]) -> list[str]:
    """Translate non-BitWAM YAML fields to the upstream Draccus CLI."""
    ignored = _RUNTIME_FIELDS | {"_config_path", "output_dir", "run_id", "seed"}
    argv = ["finetune_bitnet.py"]
    for key, value in config.items():
        if key in ignored:
            continue
        if isinstance(value, bool):
            value = "True" if value else "False"
        argv.extend((f"--{key}", str(value)))
    return argv


def _load_upstream_trainer(runtime: BitVLARuntimeConfig) -> ModuleType:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=runtime.upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != runtime.upstream_revision:
        raise ValueError(
            f"BitVLA revision mismatch: expected {runtime.upstream_revision}, found {revision}"
        )
    openvla_root = runtime.upstream_root / "openvla-oft"
    script = openvla_root / "vla-scripts/finetune_bitnet.py"
    for path in (
        str(openvla_root / "bitvla"),
        str(openvla_root),
        str(runtime.upstream_root),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(openvla_root)
    spec = importlib.util.spec_from_file_location("bitwam_upstream_bitvla_train", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load upstream trainer: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unwrap(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key.removeprefix("module."): value for key, value in state.items()}


def _install_future_frame_patch() -> None:
    import tensorflow as tf
    from prismatic.vla.datasets.rlds import traj_transforms

    assert _RUNTIME is not None
    tf.random.set_seed(_RUNTIME.seed + int(os.environ.get("RANK", "0")))
    original = traj_transforms.chunk_act_obs

    def chunk_act_obs_with_future(traj, window_size: int, future_action_window_size: int = 0):
        primary_images = traj["observation"]["image_primary"]
        trajectory_length = tf.shape(traj["action"])[0]
        result = original(traj, window_size, future_action_window_size)
        # The original chunk contains actions t..t+H but keeps examples through T-H.
        # Drop its final example so the target is the observation after all H+1 actions.
        result = tf.nest.map_structure(lambda value: value[:-1], result)
        target_indices = tf.range(future_action_window_size + 1, trajectory_length)
        result["observation"]["image_future_primary"] = tf.gather(
            primary_images, target_indices
        )[:, None]
        result["observation"]["pad_mask_dict"]["image_future_primary"] = tf.ones_like(
            result["observation"]["pad_mask"]
        )
        return result

    traj_transforms.chunk_act_obs = chunk_act_obs_with_future


def _future_transform_class(base_class):
    from PIL import Image

    class FutureBitVLATransform(base_class):
        def __call__(self, rlds_batch):
            result = super().__call__(rlds_batch)
            future_image = Image.fromarray(
                rlds_batch["observation"]["image_future_primary"][0]
            )
            result["future_pixel_value"] = self.processor.image_processor.preprocess(
                future_image, return_tensors="pt"
            )["pixel_values"][0]
            return result

    return FutureBitVLATransform


def _future_collator_class(base_class):
    class FutureBitVLACollator(base_class):
        def __call__(self, instances):
            result = super().__call__(instances)
            result["future_pixel_values"] = torch.stack(
                [instance["future_pixel_value"] for instance in instances]
            )
            return result

    return FutureBitVLACollator


def _initialize_world_head(action_head: nn.Module, device_id: int, constants: ModuleType) -> None:
    global _WORLD_HEAD
    assert _RUNTIME is not None
    input_width = int(action_head.model.layer_norm1.normalized_shape[0])
    latent_dim = input_width // int(constants.ACTION_DIM)
    world_head = LatentWorldModelHead(
        latent_dim,
        action_chunk_size=int(constants.NUM_ACTIONS_CHUNK),
        action_dim=int(constants.ACTION_DIM),
        action_embedding_dim=_RUNTIME.world_action_embedding_dim,
        hidden_dim=_RUNTIME.world_hidden_dim,
        ternary=_RUNTIME.world_head_precision == "ternary",
    ).to(device_id, dtype=torch.bfloat16)
    if _RUNTIME.world_checkpoint is not None:
        world_head.load_state_dict(_load_state(_RUNTIME.world_checkpoint))
        print(f"Loaded world-model checkpoint: {_RUNTIME.world_checkpoint}")
    _WORLD_HEAD = torch.nn.parallel.DistributedDataParallel(
        world_head,
        device_ids=[device_id],
        gradient_as_bucket_view=True,
    )
    trainable = sum(parameter.numel() for parameter in world_head.parameters())
    print(f"# trainable params in world_model: {trainable}")


def _patch_module_wrapping(upstream: ModuleType) -> None:
    original_wrap_ddp = upstream.wrap_ddp
    constants = upstream

    def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False):
        assert _RUNTIME is not None
        if module.__class__.__name__ == "L1RegressionActionHead":
            _initialize_world_head(module, device_id, constants)
        if _RUNTIME.freeze_policy:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            return _FrozenModule(module)
        return original_wrap_ddp(module, device_id, find_unused)

    upstream.wrap_ddp = wrap_ddp


def _patch_checkpoint_loading(upstream: ModuleType) -> None:
    """Allow a cross-embodiment stage to source small heads independently.

    Upstream BitVLA couples the backbone, action head, and proprio projector to
    one resume directory. DROID world/action mid-training intentionally omits
    proprio because DROID has seven state values while LIBERO has eight. The
    LIBERO post-training stage therefore restores the DROID-adapted backbone
    and action head while loading the released LIBERO proprio projector from
    its exact checkpoint.
    """
    original = upstream.load_checkpoint

    def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu"):
        assert _RUNTIME is not None
        overrides = {
            "action_head": _RUNTIME.action_checkpoint,
            "proprio_projector": _RUNTIME.proprio_checkpoint,
        }
        checkpoint = overrides.get(module_name)
        if checkpoint is None:
            return original(module_name, path, step, device)
        print(f"Loading explicit {module_name} checkpoint: {checkpoint}")
        return _load_state(checkpoint)

    upstream.load_checkpoint = load_checkpoint


def _patch_optimizer(upstream: ModuleType) -> None:
    original_adamw = upstream.AdamW

    def adamw(parameters, *args, **kwargs):
        global _OPTIMIZER
        assert _RUNTIME is not None and _WORLD_HEAD is not None
        parameters = list(parameters)
        if parameters and isinstance(parameters[0], dict):
            groups = parameters
        else:
            groups = [{"params": parameters}] if parameters else []
        groups.append(
            {
                "params": [
                    parameter for parameter in _WORLD_HEAD.parameters() if parameter.requires_grad
                ],
                "lr": _RUNTIME.world_learning_rate,
            }
        )
        optimizer = original_adamw(groups, *args, **kwargs)
        if _RUNTIME.optimizer_checkpoint is not None:
            state = torch.load(
                _RUNTIME.optimizer_checkpoint, map_location="cpu", weights_only=True
            )
            optimizer.load_state_dict(state)
            print(f"Loaded optimizer checkpoint: {_RUNTIME.optimizer_checkpoint}")
        _OPTIMIZER = optimizer
        return optimizer

    upstream.AdamW = adamw

    original_sequential_lr = upstream.torch.optim.lr_scheduler.SequentialLR

    def sequential_lr(*args, **kwargs):
        global _SCHEDULER
        scheduler = original_sequential_lr(*args, **kwargs)
        if _RUNTIME is not None and _RUNTIME.optimizer_checkpoint is not None:
            scheduler_path = Path(str(_RUNTIME.optimizer_checkpoint).replace("optimizer--", "scheduler--"))
            if scheduler_path.is_file():
                state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
                scheduler.load_state_dict(state)
                print(f"Loaded scheduler checkpoint: {scheduler_path}")
        _SCHEDULER = scheduler
        return scheduler

    upstream.torch.optim.lr_scheduler.SequentialLR = sequential_lr


def _run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_proprio=True,
):
    del action_tokenizer
    global _FORWARD_COUNT
    assert _RUNTIME is not None and _WORLD_HEAD is not None
    if not use_l1_regression:
        raise ValueError("BitVLA world post-training requires the L1 action head")

    labels = batch["labels"].to(device_id)
    actions = batch["actions"].to(device_id, dtype=torch.bfloat16)
    proprio = batch["proprio"]
    if use_proprio and proprio is not None:
        proprio = proprio.to(device_id, dtype=torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(device_id, dtype=torch.bfloat16),
            labels=labels,
            output_hidden_states=True,
            proprio=proprio if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
        )

    shifted_labels = labels[:, 1:]
    current_mask = vla.module._process_action_masks(shifted_labels)
    hidden_states = output.hidden_states[-1][:, :-1]
    batch_size = labels.shape[0]
    action_token_count = int(current_mask.sum().item()) // batch_size
    action_hidden_states = hidden_states[current_mask].reshape(batch_size, action_token_count, -1)
    predicted_actions = action_head.module.predict_action(
        action_hidden_states.to(torch.bfloat16)
    )
    action_loss = torch.nn.functional.l1_loss(actions, predicted_actions)

    model = vla.module
    future_pixels = batch["future_pixel_values"].to(device_id, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        future_features = model.get_image_features(
            pixel_values=future_pixels,
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        )
        future_latent = future_features.mean(dim=1)

    shuffled_actions = actions.roll(1, dims=0) if actions.shape[0] > 1 else None
    world_output = _WORLD_HEAD(
        action_hidden_states,
        actions,
        future_latent,
        shuffled_actions=shuffled_actions,
        contrastive_margin=_RUNTIME.world_contrastive_margin,
    )
    world_objective = world_output.loss
    if world_output.contrastive_loss is not None:
        world_objective = (
            world_objective
            + _RUNTIME.world_contrastive_weight * world_output.contrastive_loss
        )
        _METRICS["world_shuffled_action_loss"].append(
            float(world_output.shuffled_action_loss.detach())
        )
        _METRICS["world_action_conditioning_gap"].append(
            float(world_output.action_conditioning_gap.detach())
        )
        _METRICS["world_contrastive_loss"].append(
            float(world_output.contrastive_loss.detach())
        )
    loss = action_loss + _RUNTIME.world_loss_weight * world_objective
    _METRICS["world_loss"].append(float(world_output.loss.detach()))
    _METRICS["world_cosine_similarity"].append(
        float(world_output.cosine_similarity.detach())
    )
    _METRICS["action_loss"].append(float(action_loss.detach()))
    _FORWARD_COUNT += 1
    if _RUNTIME.metrics_path is not None and _FORWARD_COUNT % 10 == 0:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            _RUNTIME.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "micro_step": _FORWARD_COUNT,
                **{name: sum(values) / len(values) for name, values in _METRICS.items()},
            }
            with _RUNTIME.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload) + "\n")

    current_action_loss = torch.nn.functional.l1_loss(actions[:, 0], predicted_actions[:, 0])
    next_action_loss = torch.nn.functional.l1_loss(actions[:, 1:], predicted_actions[:, 1:])
    return loss, {
        "loss_value": float(loss.detach()),
        "curr_action_l1_loss": float(current_action_loss.detach()),
        "next_actions_l1_loss": float(next_action_loss.detach()),
    }


def _patch_metrics(upstream: ModuleType) -> None:
    original = upstream.compute_smoothened_metrics

    def compute_smoothened_metrics(metrics_deques):
        metrics = original(metrics_deques)
        metrics.update(
            {
                name: sum(values) / len(values)
                for name, values in _METRICS.items()
                if values
            }
        )
        return metrics

    upstream.compute_smoothened_metrics = compute_smoothened_metrics


def _patch_checkpointing(upstream: ModuleType) -> None:
    def save_training_checkpoint(
        cfg,
        run_dir,
        log_step,
        vla,
        processor,
        proprio_projector,
        action_head,
        train_dataset,
        distributed_state,
    ):
        assert _RUNTIME is not None and _WORLD_HEAD is not None
        if cfg.save_latest_checkpoint_only:
            checkpoint_dir = run_dir
            suffix = "latest_checkpoint.pt"
        else:
            checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
            suffix = f"{log_step}_checkpoint.pt"

        if distributed_state.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(checkpoint_dir)
            upstream.save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
            if _RUNTIME.save_backbone:
                _unwrap(vla).save_pretrained(checkpoint_dir)
            else:
                reference = {
                    "base_model": cfg.vla_path,
                    "base_step": cfg.resume_step,
                }
                (checkpoint_dir / "base_model_reference.json").write_text(
                    json.dumps(reference, indent=2) + "\n", encoding="utf-8"
                )
            if action_head is not None:
                torch.save(
                    _unwrap(action_head).state_dict(),
                    checkpoint_dir / f"action_head--{suffix}",
                )
            if proprio_projector is not None:
                torch.save(
                    _unwrap(proprio_projector).state_dict(),
                    checkpoint_dir / f"proprio_projector--{suffix}",
                )
            torch.save(
                _unwrap(_WORLD_HEAD).state_dict(),
                checkpoint_dir / f"world_model--{suffix}",
            )
            if _OPTIMIZER is not None:
                torch.save(_OPTIMIZER.state_dict(), checkpoint_dir / f"optimizer--{suffix}")
            if _SCHEDULER is not None:
                torch.save(_SCHEDULER.state_dict(), checkpoint_dir / f"scheduler--{suffix}")
            manifest = {
                "architecture": "native-bitvla-wam",
                "stage": _RUNTIME.stage,
                "upstream_revision": _RUNTIME.upstream_revision,
                "world_loss_weight": _RUNTIME.world_loss_weight,
                "world_learning_rate": _RUNTIME.world_learning_rate,
                "world_head_precision": _RUNTIME.world_head_precision,
                "world_contrastive_weight": _RUNTIME.world_contrastive_weight,
                "world_contrastive_margin": _RUNTIME.world_contrastive_margin,
                "freeze_policy": _RUNTIME.freeze_policy,
                "dataset_name": cfg.dataset_name,
                "data_root_dir": str(cfg.data_root_dir),
                "base_model": cfg.vla_path,
                "seed": _RUNTIME.seed,
                "batch_size_per_rank": cfg.batch_size,
                "gradient_accumulation_steps": cfg.grad_accumulation_steps,
                "use_proprio": cfg.use_proprio,
                "action_checkpoint": (
                    str(_RUNTIME.action_checkpoint) if _RUNTIME.action_checkpoint else None
                ),
                "proprio_checkpoint": (
                    str(_RUNTIME.proprio_checkpoint) if _RUNTIME.proprio_checkpoint else None
                ),
                "step": log_step,
            }
            (checkpoint_dir / "bitwam_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Saved BitVLA WAM checkpoint for step {log_step}: {checkpoint_dir}")
        torch.distributed.barrier()

    upstream.save_training_checkpoint = save_training_checkpoint


def install_bitwam_patches(upstream: ModuleType) -> None:
    """Install the narrow data, loss, optimizer, and checkpoint integration."""
    _install_future_frame_patch()
    upstream.BitVLA_RLDSBatchTransform = _future_transform_class(
        upstream.BitVLA_RLDSBatchTransform
    )
    upstream.Bitvla_PaddedCollatorForActionPrediction = _future_collator_class(
        upstream.Bitvla_PaddedCollatorForActionPrediction
    )
    _patch_module_wrapping(upstream)
    _patch_checkpoint_loading(upstream)
    _patch_optimizer(upstream)
    _patch_metrics(upstream)
    _patch_checkpointing(upstream)
    upstream.run_forward_pass = _run_forward_pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    global _RUNTIME
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if config.get("architecture") != "bitvla":
        raise ValueError("BitVLA training configs must set architecture: bitvla")
    _RUNTIME = parse_runtime_config(config)
    os.environ["PYTHONHASHSEED"] = str(_RUNTIME.seed)
    random.seed(_RUNTIME.seed)
    np.random.seed(_RUNTIME.seed)
    torch.manual_seed(_RUNTIME.seed)
    upstream = _load_upstream_trainer(_RUNTIME)
    install_bitwam_patches(upstream)
    sys.argv = build_upstream_argv(config)
    upstream.finetune()


if __name__ == "__main__":
    main()
