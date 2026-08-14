"""Local LeRobot training entrypoint with streaming-safe distributed batches."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate


def configure_streaming_dataloader(accelerator: Any, *, streaming: bool) -> Any:
    """Let each rank fetch its own iterable batch so language strings stay as Python lists."""
    if streaming:
        accelerator.dataloader_config.dispatch_batches = False
    return accelerator


@torch.no_grad()
def clip_cpu_offloaded_grad_norm(
    accelerator: Any,
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Clip CPU-offloaded FSDP2 shards using a GPU scalar collective."""
    if norm_type != 2.0:
        raise ValueError("BitWAM's CPU-offloaded FSDP clipper supports only L2 norms")
    accelerator.unscale_gradients()
    parameters = list(parameters)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_squared_norm = torch.zeros((), dtype=torch.float64)

    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if isinstance(gradient, DTensor):
            local_gradient = gradient.to_local()
            replicated = all(isinstance(placement, Replicate) for placement in gradient.placements)
            if replicated and rank != 0:
                continue
        else:
            local_gradient = gradient
            if world_size > 1 and rank != 0:
                continue
        local_squared_norm.add_(local_gradient.detach().double().pow(2).sum().cpu())

    squared_norm = local_squared_norm.to(accelerator.device)
    if dist.is_initialized():
        dist.all_reduce(squared_norm, op=dist.ReduceOp.SUM)
    total_norm = squared_norm.sqrt()
    clip_coefficient = (float(max_norm) / (total_norm + 1e-6)).clamp(max=1.0)
    coefficient = clip_coefficient.cpu()

    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        local_gradient = gradient.to_local() if isinstance(gradient, DTensor) else gradient
        local_gradient.mul_(coefficient.to(dtype=local_gradient.dtype))
    return total_norm


def _install_streaming_patch() -> None:
    import lerobot.distributed as distributed
    from lerobot.distributed import factory

    original = factory.make_accelerator

    def make_accelerator(cfg):
        accelerator = original(cfg)
        accelerator = configure_streaming_dataloader(
            accelerator,
            streaming=cfg.dataset.streaming,
        )
        if cfg.parallelism.is_sharded and cfg.accelerator.fsdp.cpu_offload:
            accelerator.clip_grad_norm_ = MethodType(
                clip_cpu_offloaded_grad_norm,
                accelerator,
            )
        return accelerator

    factory.make_accelerator = make_accelerator
    distributed.make_accelerator = make_accelerator


def main() -> None:
    """Install the narrow compatibility patch before importing the pinned trainer."""
    _install_streaming_patch()
    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()


if __name__ == "__main__":
    main()
