"""Activate BitVLA's native two-bit inference representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PackingReport:
    """Persistent-storage accounting for converted BitLinear matrices."""

    packed_layers: int
    packed_weight_count: int
    bf16_weight_bytes_replaced: int
    packed_weight_bytes: int

    @property
    def weight_storage_reduction(self) -> float:
        if self.bf16_weight_bytes_replaced == 0:
            return 0.0
        return 1 - self.packed_weight_bytes / self.bf16_weight_bytes_replaced

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self) | {"weight_storage_reduction": self.weight_storage_reduction}


@torch.inference_mode()
def pack_bitlinear_weights(module: nn.Module) -> PackingReport:
    """Replace eligible BitLinear BF16 masters with their upstream uint8 2-bit buffers."""
    packed_layers = packed_weight_count = 0
    bf16_weight_bytes_replaced = packed_weight_bytes = 0
    for candidate in module.modules():
        if candidate.__class__.__name__ != "BitLinear":
            continue
        quantize_weights = getattr(candidate, "quantize_weights", None)
        weight = getattr(candidate, "weight", None)
        weight_bits = int(getattr(candidate, "weight_bits", 1))
        if not callable(quantize_weights) or not isinstance(weight, torch.Tensor) or weight_bits != 1:
            continue

        original_count = weight.numel()
        original_bytes = original_count * weight.element_size()
        quantize_weights()
        packed = getattr(candidate, "q_weight", None)
        if getattr(candidate, "weight", None) is not None or not isinstance(packed, torch.Tensor):
            raise RuntimeError(f"BitLinear packing did not replace the BF16 weight: {candidate!r}")
        if packed.dtype != torch.uint8:
            raise RuntimeError(f"BitLinear packing produced {packed.dtype}, expected torch.uint8")
        if packed.numel() != (original_count + 3) // 4:
            raise RuntimeError("BitLinear packing did not store exactly four ternary codes per byte")

        packed_layers += 1
        packed_weight_count += original_count
        bf16_weight_bytes_replaced += original_bytes
        packed_weight_bytes += packed.numel() * packed.element_size()

    if packed_layers == 0:
        raise RuntimeError("No eligible one-bit BitLinear layers were found")
    return PackingReport(
        packed_layers=packed_layers,
        packed_weight_count=packed_weight_count,
        bf16_weight_bytes_replaced=bf16_weight_bytes_replaced,
        packed_weight_bytes=packed_weight_bytes,
    )
