"""Activate BitVLA's native two-bit inference representation."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PackingReport:
    """Persistent-storage accounting for converted BitLinear matrices."""

    scope: str
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


def dequantize_packed_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Decode four two-bit ternary codes per byte into a BF16 matrix."""
    codes = torch.stack(
        (
            packed & 0x03,
            (packed >> 2) & 0x03,
            (packed >> 4) & 0x03,
            (packed >> 6) & 0x03,
        ),
        dim=1,
    ).flatten()[: rows * columns]
    return ((codes.float() - 1) * scale).to(torch.bfloat16).view(rows, columns)


def unpack_packed_weight_int8_transposed(
    packed: torch.Tensor,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Decode a packed row-major matrix directly into contiguous K-by-N INT8 layout."""
    positions = torch.arange(rows * columns, device=packed.device)
    packed_values = packed[positions // 4]
    shifts = (positions % 4) * 2
    codes = ((packed_values >> shifts) & 0x03).to(torch.int8) - 1
    return codes.view(rows, columns).T.contiguous()


def quantize_activation_int8(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token INT8 activations and the inverse quantization scale."""
    inverse_scale = inputs.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127
    quantized = (inputs.float() / inverse_scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, inverse_scale


def quantize_activation(inputs: torch.Tensor) -> torch.Tensor:
    """Apply BitVLA's per-token eight-bit activation fake quantization."""
    dtype = inputs.dtype
    values = inputs.float()
    scale = 127 / values.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    return ((values * scale).round().clamp(-128, 127) / scale).to(dtype)


def packed_linear(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Decode a packed matrix and apply its linear projection."""
    weight = dequantize_packed_weight(packed, scale, rows, columns)
    return F.linear(inputs, weight, bias)


def _packed_forward(
    layer: nn.Module,
    inputs: torch.Tensor,
    compiled_dequantize: Any,
    compiled_activation_quant: Any | None = None,
) -> torch.Tensor:
    if int(getattr(layer, "input_bits", 8)) == 8:
        if compiled_activation_quant is None:
            upstream_module = sys.modules[layer.__class__.__module__]
            inputs = upstream_module.ActQuant.apply(inputs)
        else:
            inputs = compiled_activation_quant(inputs)
    rows, columns = layer.orig_shape
    weight = compiled_dequantize(layer.q_weight, layer.w_step, rows, columns)
    return F.linear(inputs, weight, layer.bias)


def _torch_int8_forward(
    layer: nn.Module,
    inputs: torch.Tensor,
    compiled_unpack: Any,
    compiled_activation_quant: Any,
) -> torch.Tensor:
    input_shape = inputs.shape
    rows, columns = layer.orig_shape
    quantized_inputs, inverse_scale = compiled_activation_quant(inputs.reshape(-1, columns))
    weight_transposed = compiled_unpack(layer.q_weight, rows, columns)
    accumulated = torch._int_mm(quantized_inputs, weight_transposed)
    output = accumulated.float() * inverse_scale * layer.w_step.float()
    output = output.to(inputs.dtype).view(*input_shape[:-1], rows)
    if layer.bias is not None:
        output = output + layer.bias
    return output


def _triton_direct_forward(
    layer: nn.Module,
    inputs: torch.Tensor,
    *,
    activation_backend: str,
    bf16_candidate: bool,
) -> torch.Tensor:
    from lerobot_policy_bitwam.bitvla_packed_kernel import (
        direct_packed_bf16_linear,
        direct_packed_int8_linear,
    )

    rows, columns = layer.orig_shape
    if bf16_candidate:
        return direct_packed_bf16_linear(
            inputs,
            layer.q_weight,
            layer.w_step,
            layer.bias,
            rows,
            columns,
            backend="triton",
            activation_backend=activation_backend,
        )
    return direct_packed_int8_linear(
        inputs,
        layer.q_weight,
        layer.w_step,
        layer.bias,
        rows,
        columns,
        backend="triton",
        activation_backend=activation_backend,
        decode_mode="scalar",
    )


def _direct_w2a8_forward(
    layer: nn.Module,
    inputs: torch.Tensor,
    *,
    activation_backend: str,
    small_batch_threshold: int,
) -> torch.Tensor:
    from lerobot_policy_bitwam.bitvla_packed_kernel import direct_packed_w2a8_linear

    rows, columns = layer.orig_shape
    return direct_packed_w2a8_linear(
        inputs,
        layer.q_weight,
        layer.w_step,
        layer.bias,
        rows,
        columns,
        backend="triton",
        activation_backend=activation_backend,
        small_batch_threshold=small_batch_threshold,
    )


def _cached_dense_forward(layer: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run the upstream activation quantizer against a pre-quantized BF16 weight."""
    upstream_module = sys.modules[layer.__class__.__module__]
    if int(getattr(layer, "input_bits", 8)) == 8:
        inputs = upstream_module.ActQuant.apply(inputs)
    return F.linear(inputs, layer.weight, layer.bias)


def _compiled_linear_forward(layer: nn.Module, inputs: torch.Tensor, compiled_linear: Any) -> torch.Tensor:
    upstream_module = sys.modules[layer.__class__.__module__]
    if int(getattr(layer, "input_bits", 8)) == 8:
        inputs = upstream_module.ActQuant.apply(inputs)
    rows, columns = layer.orig_shape
    return compiled_linear(inputs, layer.q_weight, layer.w_step, layer.bias, rows, columns)


def enable_compiled_bitlinear_unpack(module: nn.Module) -> int:
    """Fuse packed-weight decoding into one generated CUDA kernel per matrix shape."""
    compiled_dequantize = torch.compile(dequantize_packed_weight, fullgraph=True, dynamic=False)
    optimized_layers = 0
    for candidate in module.modules():
        if candidate.__class__.__name__ != "BitLinear" or not bool(getattr(candidate, "enable_qlora", False)):
            continue
        if not isinstance(getattr(candidate, "q_weight", None), torch.Tensor):
            continue
        candidate.w_step = candidate.w_step.to(candidate.q_weight.device)
        candidate.forward = MethodType(
            lambda layer, inputs, compiled=compiled_dequantize: _packed_forward(layer, inputs, compiled),
            candidate,
        )
        optimized_layers += 1
    if optimized_layers == 0:
        raise RuntimeError("No packed BitLinear layers were available for compiled unpacking")
    return optimized_layers


def enable_compiled_bitlinear_runtime(module: nn.Module) -> int:
    """Fuse both activation quantization and packed-weight decoding for inference."""
    compiled_dequantize = torch.compile(dequantize_packed_weight, fullgraph=True, dynamic=False)
    compiled_activation_quant = torch.compile(quantize_activation, fullgraph=True, dynamic=True)
    optimized_layers = 0
    for candidate in module.modules():
        if candidate.__class__.__name__ != "BitLinear" or not bool(getattr(candidate, "enable_qlora", False)):
            continue
        if not isinstance(getattr(candidate, "q_weight", None), torch.Tensor):
            continue
        candidate.w_step = candidate.w_step.to(candidate.q_weight.device)
        candidate.forward = MethodType(
            lambda layer,
            inputs,
            dequantize=compiled_dequantize,
            activation=(compiled_activation_quant): _packed_forward(layer, inputs, dequantize, activation),
            candidate,
        )
        optimized_layers += 1
    if optimized_layers == 0:
        raise RuntimeError("No packed BitLinear layers were available for the compiled runtime")
    return optimized_layers


def enable_compiled_bitlinear_projection(module: nn.Module) -> int:
    """Compile packed decoding together with the BF16 projection, preserving activation math."""
    compiled_linear = torch.compile(packed_linear, fullgraph=True, dynamic=True)
    optimized_layers = 0
    for candidate in module.modules():
        if candidate.__class__.__name__ != "BitLinear" or not bool(getattr(candidate, "enable_qlora", False)):
            continue
        if not isinstance(getattr(candidate, "q_weight", None), torch.Tensor):
            continue
        candidate.w_step = candidate.w_step.to(candidate.q_weight.device)
        candidate.forward = MethodType(
            lambda layer, inputs, projection=compiled_linear: _compiled_linear_forward(
                layer, inputs, projection
            ),
            candidate,
        )
        optimized_layers += 1
    if optimized_layers == 0:
        raise RuntimeError("No packed BitLinear layers were available for compiled projections")
    return optimized_layers


def enable_torch_int8_bitlinear_runtime(module: nn.Module) -> int:
    """Use packed INT2 weights with native CUDA INT8 tensor-core matrix multiplication."""
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("The installed PyTorch build does not provide CUDA INT8 matrix multiply")
    compiled_unpack = torch.compile(
        unpack_packed_weight_int8_transposed,
        fullgraph=True,
        dynamic=False,
    )
    compiled_activation_quant = torch.compile(
        quantize_activation_int8,
        fullgraph=True,
        dynamic=False,
    )
    optimized_layers = 0
    for candidate in module.modules():
        if candidate.__class__.__name__ != "BitLinear" or not bool(getattr(candidate, "enable_qlora", False)):
            continue
        if not isinstance(getattr(candidate, "q_weight", None), torch.Tensor):
            continue
        candidate.w_step = candidate.w_step.to(candidate.q_weight.device)
        candidate.forward = MethodType(
            lambda layer, inputs, unpack=compiled_unpack, activation=compiled_activation_quant: (
                _torch_int8_forward(layer, inputs, unpack, activation)
            ),
            candidate,
        )
        optimized_layers += 1
    if optimized_layers == 0:
        raise RuntimeError("No packed BitLinear layers were available for the INT8 runtime")
    return optimized_layers


@torch.inference_mode()
def enable_cached_dense_bitlinear_runtime(module: nn.Module, *, scope: str = "all") -> int:
    """Cache ternary weights as BF16 for a fair optimized dense baseline.

    Native BitVLA applies ``WeightQuant`` to every dense master matrix on every
    inference call.  That is correct for training but needlessly expensive for
    deployment.  This baseline applies the exact upstream quantizer once,
    replaces the master value in-place with the resulting BF16 matrix, and
    keeps activation quantization unchanged.  It therefore removes recurrent
    weight-quantization work without receiving any of the packed runtime's
    storage advantage.
    """
    optimized_layers = 0
    for candidate in module.modules():
        if candidate.__class__.__name__ != "BitLinear":
            continue
        weight = getattr(candidate, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or int(getattr(candidate, "weight_bits", 1)) != 1
            or not _matches_scope(candidate, scope)
        ):
            continue
        upstream_module = sys.modules[candidate.__class__.__module__]
        weight_quantizer = getattr(upstream_module, "WeightQuant", None)
        if weight_quantizer is None or not callable(getattr(weight_quantizer, "apply", None)):
            raise RuntimeError(
                f"BitLinear module {candidate.__class__.__module__!r} has no WeightQuant.apply"
            )
        cached_weight = weight_quantizer.apply(weight).detach().contiguous()
        if cached_weight.dtype != weight.dtype or cached_weight.shape != weight.shape:
            raise RuntimeError("Cached dense ternary weight changed dtype or shape")
        candidate.weight = nn.Parameter(cached_weight, requires_grad=False)
        candidate.forward = MethodType(_cached_dense_forward, candidate)
        candidate.cached_dense_ternary = True
        optimized_layers += 1
    if optimized_layers == 0:
        raise RuntimeError("No eligible one-bit BitLinear layers were found for dense caching")
    return optimized_layers


def enable_triton_direct_bitlinear_runtime(
    module: nn.Module,
    *,
    activation_backend: str = "torch",
    bf16_candidate: bool = False,
) -> int:
    """Use the experimental direct packed Triton projection without a dense weight tensor.

    The caller must apply an end-to-end action equality gate before latency or
    policy-quality measurements.  ``activation_backend='torch'`` preserves the
    upstream activation quantization operation order; the integer accumulation
    itself is not equivalent to upstream BF16 ``F.linear``.
    """
    from lerobot_policy_bitwam.bitvla_packed_kernel import triton_packed_kernel_available

    if not triton_packed_kernel_available():
        raise RuntimeError("The direct packed runtime requires Triton and CUDA")
    if activation_backend not in {"torch", "hybrid", "triton"}:
        raise ValueError(f"Unsupported direct activation backend: {activation_backend}")

    optimized_layers = 0
    for candidate in module.modules():
        if candidate.__class__.__name__ != "BitLinear" or not bool(
            getattr(candidate, "enable_qlora", False)
        ):
            continue
        if not isinstance(getattr(candidate, "q_weight", None), torch.Tensor):
            continue
        candidate.w_step = candidate.w_step.to(candidate.q_weight.device)
        candidate.forward = MethodType(
            lambda layer,
            inputs,
            activation=activation_backend,
            bf16=bf16_candidate: _triton_direct_forward(
                layer,
                inputs,
                activation_backend=activation,
                bf16_candidate=bf16,
            ),
            candidate,
        )
        optimized_layers += 1
    if optimized_layers == 0:
        raise RuntimeError("No packed BitLinear layers were available for direct Triton inference")
    return optimized_layers


@torch.inference_mode()
def repack_bitlinear_weights_for_dp4a(module: nn.Module) -> int:
    """Replace row-major two-bit buffers with equally compact DP4A buffers."""
    from lerobot_policy_bitwam.bitvla_packed_kernel import repack_packed_ternary_for_dp4a

    prepared_layers = 0
    for candidate in module.modules():
        if candidate.__class__.__name__ != "BitLinear":
            continue
        packed = getattr(candidate, "q_weight", None)
        if not isinstance(packed, torch.Tensor):
            continue
        rows, columns = candidate.orig_shape
        if columns % 16:
            raise RuntimeError(
                f"Direct W2A8 requires input width divisible by 16, got {candidate.orig_shape}"
            )
        layout = str(getattr(candidate, "packed_layout", "row_major"))
        if layout == "row_major":
            candidate.q_weight = repack_packed_ternary_for_dp4a(packed, rows, columns)
            candidate.packed_layout = "dp4a"
        elif layout != "dp4a":
            raise RuntimeError(f"Unsupported packed BitLinear layout: {layout}")
        prepared_layers += 1
    if prepared_layers == 0:
        raise RuntimeError("No packed BitLinear layers were available for DP4A repacking")
    return prepared_layers


@torch.inference_mode()
def enable_direct_w2a8_bitlinear_runtime(
    module: nn.Module,
    *,
    activation_backend: str = "hybrid",
    small_batch_threshold: int = 8,
) -> int:
    """Enable the production direct packed W2A8 runtime.

    Every eligible layer is transformed packed-to-packed into the interleaved
    DP4A layout.  The original buffer is replaced, so storage remains exactly
    two bits per weight.  Small token batches use the DP4A mainloop while
    larger prompt/image batches consume the same buffer with INT8 Tensor Cores.
    No dense BF16 or INT8 weight cache is created at setup or during forward.
    """
    from lerobot_policy_bitwam.bitvla_packed_kernel import (
        triton_packed_kernel_available,
    )

    if not triton_packed_kernel_available():
        raise RuntimeError("The direct W2A8 runtime requires Triton and CUDA")
    if activation_backend not in {"torch", "hybrid", "triton"}:
        raise ValueError(f"Unsupported direct activation backend: {activation_backend}")
    if small_batch_threshold < 1:
        raise ValueError("small_batch_threshold must be positive")

    prepared_layers = repack_bitlinear_weights_for_dp4a(module)
    optimized_layers = 0
    for candidate in module.modules():
        if candidate.__class__.__name__ != "BitLinear" or not bool(
            getattr(candidate, "enable_qlora", False)
        ):
            continue
        packed = getattr(candidate, "q_weight", None)
        if not isinstance(packed, torch.Tensor):
            continue
        if int(getattr(candidate, "input_bits", 8)) != 8:
            raise RuntimeError("Direct W2A8 requires eight-bit BitLinear activations")
        layout = str(getattr(candidate, "packed_layout", "row_major"))
        if layout != "dp4a":
            raise RuntimeError(f"Unsupported packed BitLinear layout: {layout}")
        candidate.w_step = candidate.w_step.to(candidate.q_weight.device)
        candidate.forward = MethodType(
            lambda layer,
            inputs,
            activation=activation_backend,
            threshold=small_batch_threshold: _direct_w2a8_forward(
                layer,
                inputs,
                activation_backend=activation,
                small_batch_threshold=threshold,
            ),
            candidate,
        )
        optimized_layers += 1
    if optimized_layers == 0:
        raise RuntimeError("No packed BitLinear layers were available for direct W2A8 inference")
    if optimized_layers != prepared_layers:
        raise RuntimeError("DP4A repacking and runtime layer counts did not match")
    return optimized_layers


def _matches_scope(candidate: nn.Module, scope: str) -> bool:
    module_name = candidate.__class__.__module__
    if scope == "all":
        return True
    if scope == "text":
        return "modeling_bitnet" in module_name
    if scope == "vision":
        return "modeling_siglip" in module_name
    weight = getattr(candidate, "weight", None)
    shape = tuple(weight.shape) if isinstance(weight, torch.Tensor) else tuple(
        getattr(candidate, "orig_shape", ())
    )
    if scope == "text_mlp":
        return "modeling_bitnet" in module_name and 6912 in shape
    if scope == "text_mlp_down":
        return "modeling_bitnet" in module_name and shape == (2560, 6912)
    if scope == "text_mlp_gate_up":
        return "modeling_bitnet" in module_name and shape == (6912, 2560)
    if scope == "text_attention":
        return "modeling_bitnet" in module_name and len(shape) == 2 and 6912 not in shape
    raise ValueError(f"Unsupported BitLinear packing scope: {scope}")


@torch.inference_mode()
def pack_bitlinear_weights(module: nn.Module, *, scope: str = "all") -> PackingReport:
    """Replace eligible BitLinear BF16 masters with their upstream uint8 2-bit buffers."""
    packed_layers = packed_weight_count = 0
    bf16_weight_bytes_replaced = packed_weight_bytes = 0
    for candidate in module.modules():
        if candidate.__class__.__name__ != "BitLinear":
            continue
        quantize_weights = getattr(candidate, "quantize_weights", None)
        weight = getattr(candidate, "weight", None)
        weight_bits = int(getattr(candidate, "weight_bits", 1))
        if (
            not callable(quantize_weights)
            or not isinstance(weight, torch.Tensor)
            or weight_bits != 1
            or not _matches_scope(candidate, scope)
        ):
            continue

        original_shape = tuple(int(value) for value in weight.shape)
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
        candidate.orig_shape = original_shape
        candidate.n_elems = original_count
        candidate.enable_qlora = True
        candidate.packed_layout = "row_major"

        packed_layers += 1
        packed_weight_count += original_count
        bf16_weight_bytes_replaced += original_bytes
        packed_weight_bytes += packed.numel() * packed.element_size()

    if packed_layers == 0:
        raise RuntimeError("No eligible one-bit BitLinear layers were found")
    return PackingReport(
        scope=scope,
        packed_layers=packed_layers,
        packed_weight_count=packed_weight_count,
        bf16_weight_bytes_replaced=bf16_weight_bytes_replaced,
        packed_weight_bytes=packed_weight_bytes,
    )
