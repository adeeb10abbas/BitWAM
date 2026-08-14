"""Tests for the narrow distributed streaming compatibility shim."""

from types import SimpleNamespace

import pytest
import torch

from lerobot_policy_bitwam.train_entrypoint import (
    clip_cpu_offloaded_grad_norm,
    configure_streaming_dataloader,
)


class _CPUAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def unscale_gradients() -> None:
        pass


def test_streaming_disables_central_batch_dispatch() -> None:
    accelerator = SimpleNamespace(dataloader_config=SimpleNamespace(dispatch_batches=None))
    assert configure_streaming_dataloader(accelerator, streaming=True) is accelerator
    assert accelerator.dataloader_config.dispatch_batches is False


def test_map_style_loader_keeps_accelerate_default() -> None:
    accelerator = SimpleNamespace(dataloader_config=SimpleNamespace(dispatch_batches=None))
    configure_streaming_dataloader(accelerator, streaming=False)
    assert accelerator.dataloader_config.dispatch_batches is None


def test_cpu_offloaded_clipper_preserves_l2_clipping() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    norm = clip_cpu_offloaded_grad_norm(_CPUAccelerator(), [parameter], 1.0)
    assert norm.item() == pytest.approx(5.0)
    assert parameter.grad.tolist() == pytest.approx([0.6, 0.8])


def test_cpu_offloaded_clipper_rejects_non_l2_norms() -> None:
    with pytest.raises(ValueError, match="only L2"):
        clip_cpu_offloaded_grad_norm(_CPUAccelerator(), [], 1.0, norm_type=1.0)
