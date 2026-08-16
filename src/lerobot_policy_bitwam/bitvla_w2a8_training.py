"""Kernel-matched straight-through training for BitWAM W2A8 projections.

Native BitVLA fake-quantizes activations and weights back to BF16 before its
linear layer.  Direct packed inference instead accumulates the INT8 activation
codes and ternary weight codes in INT32, then applies scales once.  Those two
operation orders are close but not identical.  This module makes recovery/QAT
see the deployment operation order while preserving straight-through gradients
for the BF16 master weights.
"""

from __future__ import annotations

from types import MethodType

import torch
from torch import nn

from lerobot_policy_bitwam.bitvla_packed_kernel import (
    bitvla_quantize_activation_int8,
    hybrid_bitvla_quantize_activation_int8,
    triton_bitvla_quantize_activation_int8,
)
from lerobot_policy_bitwam.bitvla_packing import _matches_scope


def _integer_matmul(activations: torch.Tensor, weights_transposed: torch.Tensor) -> torch.Tensor:
    """Return exact INT32 accumulation, using native CUDA INT8 MM when available."""
    if activations.dtype != torch.int8 or weights_transposed.dtype != torch.int8:
        raise TypeError("integer matmul requires INT8 operands")
    if activations.device.type == "cuda" and hasattr(torch, "_int_mm"):
        try:
            return torch._int_mm(activations.contiguous(), weights_transposed.contiguous())
        except RuntimeError:
            # Some CUDA builds restrict small/non-aligned shapes.  FP32 exactly
            # represents every product and complete BitVLA-sized integer sum.
            pass
    if activations.device.type == "cpu":
        return activations.to(torch.int32).matmul(weights_transposed.to(torch.int32))
    return activations.float().matmul(weights_transposed.float()).to(torch.int32)


class _W2A8LinearStraightThrough(torch.autograd.Function):
    """INT32 forward with the gradient of a fake-quantized linear projection."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation_backend: str,
    ) -> torch.Tensor:
        if inputs.shape[-1] != weight.shape[1]:
            raise ValueError("input and weight feature dimensions do not match")
        if activation_backend == "torch":
            quantized_inputs, activation_scale = bitvla_quantize_activation_int8(inputs)
        elif activation_backend == "hybrid":
            quantized_inputs, activation_scale = hybrid_bitvla_quantize_activation_int8(inputs)
        elif activation_backend == "triton":
            quantized_inputs, activation_scale = triton_bitvla_quantize_activation_int8(inputs)
        else:
            raise ValueError(f"Unsupported W2A8 QAT activation backend: {activation_backend}")
        weight_values = weight.float()
        weight_step = weight_values.abs().mean().clamp(min=1e-5)
        quantized_weight = (weight_values / weight_step).round().clamp(-1, 1).to(torch.int8)

        flat_inputs = quantized_inputs.reshape(-1, weight.shape[1])
        accumulated = _integer_matmul(flat_inputs, quantized_weight.T)
        output = accumulated.float() * activation_scale.reshape(-1, 1).reciprocal()
        output.mul_(weight_step)
        if bias is not None:
            output.add_(bias.float())

        # The backward pass is the exact gradient of F.linear evaluated at the
        # fake-quantized BF16 operands, with the quantizers treated as identity.
        dequantized_inputs = (quantized_inputs.float() / activation_scale).to(inputs.dtype)
        dequantized_weight = (quantized_weight.float() * weight_step).to(weight.dtype)
        ctx.save_for_backward(dequantized_inputs, dequantized_weight)
        ctx.input_shape = tuple(inputs.shape)
        ctx.input_dtype = inputs.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        return output.to(inputs.dtype).view(*inputs.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dequantized_inputs, dequantized_weight = ctx.saved_tensors
        flat_gradient = grad_output.reshape(-1, dequantized_weight.shape[0]).float()
        flat_inputs = dequantized_inputs.reshape(-1, dequantized_weight.shape[1]).float()
        grad_input = flat_gradient.matmul(dequantized_weight.float()).to(ctx.input_dtype)
        grad_weight = flat_gradient.T.matmul(flat_inputs).to(ctx.weight_dtype)
        grad_bias = None
        if ctx.bias_dtype is not None:
            grad_bias = flat_gradient.sum(dim=0).to(ctx.bias_dtype)
        return grad_input.view(ctx.input_shape), grad_weight, grad_bias, None


def w2a8_ste_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation_backend: str = "torch",
) -> torch.Tensor:
    """Apply the deployment W2A8 operation order with straight-through gradients."""
    return _W2A8LinearStraightThrough.apply(inputs, weight, bias, activation_backend)


def _w2a8_qat_forward(
    layer: nn.Module,
    inputs: torch.Tensor,
    *,
    activation_backend: str,
) -> torch.Tensor:
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError("W2A8 QAT requires an unpacked BF16 master weight")
    return w2a8_ste_linear(
        inputs,
        weight,
        layer.bias,
        activation_backend=activation_backend,
    )


def enable_w2a8_qat_semantics(
    module: nn.Module,
    *,
    scope: str = "all",
    activation_backend: str = "torch",
    require_layers: bool = True,
) -> int:
    """Patch eligible dense BitLinear modules to use the deployment forward contract."""
    if activation_backend not in {"torch", "hybrid", "triton"}:
        raise ValueError(f"Unsupported W2A8 QAT activation backend: {activation_backend}")
    converted = 0
    for candidate in module.modules():
        if candidate.__class__.__name__ != "BitLinear":
            continue
        if int(getattr(candidate, "weight_bits", 1)) != 1 or not _matches_scope(candidate, scope):
            continue
        if int(getattr(candidate, "input_bits", 8)) != 8:
            raise RuntimeError("W2A8 QAT requires eight-bit BitLinear activations")
        if not isinstance(getattr(candidate, "weight", None), torch.Tensor):
            raise RuntimeError("W2A8 QAT cannot start from an inference-packed BitLinear")
        candidate.forward = MethodType(
            lambda layer, inputs, backend=activation_backend: _w2a8_qat_forward(
                layer,
                inputs,
                activation_backend=backend,
            ),
            candidate,
        )
        candidate.w2a8_qat_semantics = True
        candidate.w2a8_qat_activation_backend = activation_backend
        converted += 1
    if converted == 0 and require_layers:
        raise RuntimeError("No eligible one-bit BitLinear layers were found for W2A8 QAT")
    return converted
