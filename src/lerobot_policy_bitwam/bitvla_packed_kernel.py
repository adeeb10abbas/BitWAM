"""Direct packed-ternary INT8 projection kernels for native BitVLA.

The experimental kernels deliberately never create a dense BF16 or INT8 weight
matrix.  They read BitVLA's four two-bit codes per byte from global memory,
decode a tile to ternary values in registers, and multiply that tile by
already-quantized INT8 activations.  Activations reproduce upstream
``ActQuant`` exactly: one FP32 absolute-max scale per input token,
round-to-nearest, saturation to the signed INT8 range, followed by BF16
dequantization when the BF16 exact-candidate path is selected.  The default
activation backend is the exact upstream PyTorch expression; the all-Triton
activation prepass is retained only as an explicitly approximate experiment.

The result is mathematically equivalent to the integer projection
``(q_x @ ternary_weight.T) * input_step * weight_step``.  It is intentionally
kept separate from the BitLinear monkey-patching code until CUDA correctness
and end-to-end action parity have passed their gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

try:  # Triton is a Linux/CUDA runtime dependency, not a macOS test dependency.
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - exercised on hosts without Triton.
    triton = None
    tl = None
    libdevice = None


_REFERENCE_ROW_TILE: Final = 128


@dataclass(frozen=True)
class PackedKernelConfig:
    """A shape-specialized direct packed GEMM launch configuration."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int

    def validate(self) -> None:
        for name, value in (
            ("block_m", self.block_m),
            ("block_n", self.block_n),
            ("block_k", self.block_k),
            ("num_warps", self.num_warps),
        ):
            if value < 1 or value & (value - 1):
                raise ValueError(f"{name} must be a positive power of two")
        if self.block_k < 16:
            raise ValueError("block_k must be at least 16 for Tensor Core dot")


def _select_packed_int8_config(tokens: int, rows: int, columns: int) -> PackedKernelConfig:
    """Favor output/sequence tiles that reuse direct-decoded packed weights."""
    if tokens >= 128 and rows >= 128 and columns >= 64:
        return PackedKernelConfig(block_m=64, block_n=128, block_k=64, num_warps=8)
    if tokens >= 32 and rows >= 128 and columns >= 64:
        return PackedKernelConfig(block_m=32, block_n=128, block_k=64, num_warps=8)
    if tokens >= 16 and rows >= 64 and columns >= 64:
        return PackedKernelConfig(block_m=16, block_n=128, block_k=64, num_warps=4)
    return PackedKernelConfig(block_m=8, block_n=32, block_k=32, num_warps=4)


def bitvla_quantize_activation_int8(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return upstream-ActQuant codes and its original FP32 multiplication scale.

    This intentionally retains ``127 / absmax`` instead of replacing it with
    ``absmax / 127``.  The former is BitVLA's exact operation order, including
    which side of an INT8 rounding boundary an FP32 input falls on.
    """
    values = inputs.float()
    scale = 127 / values.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    quantized = (values * scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def triton_packed_kernel_available() -> bool:
    """Return whether this interpreter can launch the direct CUDA kernel."""
    return triton is not None and torch.cuda.is_available()


def _validate_packed_projection(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
) -> None:
    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    if inputs.ndim < 1 or inputs.shape[-1] != columns:
        raise ValueError(f"inputs must end in {columns} features, got {tuple(inputs.shape)}")
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed weight must be torch.uint8, got {packed.dtype}")
    if packed.numel() != (rows * columns + 3) // 4:
        raise ValueError("packed weight does not contain exactly four two-bit codes per byte")
    if scale.numel() not in (1, rows):
        raise ValueError(f"weight scale must be scalar or have one value per output row ({rows})")
    if bias is not None and bias.numel() != rows:
        raise ValueError(f"bias must have one value per output row ({rows})")
    tensors = (packed, scale) if bias is None else (packed, scale, bias)
    if any(tensor.device != inputs.device for tensor in tensors):
        raise ValueError("inputs, packed weight, scale, and bias must share a device")


def _weight_scale_rows(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Return a flat scale view without expanding a per-channel copy."""
    if scale.numel() == 1:
        return scale.reshape(1)
    if scale.numel() != rows:
        raise ValueError(f"weight scale must be scalar or have one value per output row ({rows})")
    return scale.reshape(rows)


def packed_ternary_int8_linear_reference(
    quantized_inputs: torch.Tensor,
    input_inverse_scale: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Portable tiled reference without materializing a full decoded weight.

    ``quantized_inputs`` is ``[..., columns]`` INT8 and
    ``input_inverse_scale`` is ``[..., 1]`` FP32.  Decoded ternary values exist
    only for a small output-row tile, which makes this useful as a correctness
    oracle for the Triton implementation as well as for CPU tests.
    """
    _validate_packed_projection(
        quantized_inputs,
        packed,
        weight_scale,
        bias,
        rows,
        columns,
    )
    if quantized_inputs.dtype != torch.int8:
        raise TypeError(f"quantized inputs must be torch.int8, got {quantized_inputs.dtype}")
    if input_inverse_scale.shape != (*quantized_inputs.shape[:-1], 1):
        raise ValueError("input_inverse_scale must have shape inputs.shape[:-1] + (1,)")

    input_shape = quantized_inputs.shape
    flat_inputs = quantized_inputs.reshape(-1, columns)
    flat_input_scale = input_inverse_scale.reshape(-1, 1)
    output = torch.empty((flat_inputs.shape[0], rows), device=flat_inputs.device, dtype=torch.float32)
    flat_weight_scale = _weight_scale_rows(weight_scale, rows).to(dtype=torch.float32)

    # The byte/codes arithmetic intentionally follows BitVLA's flattened,
    # row-major packing convention, including matrices whose columns are not a
    # multiple of four.
    for row_start in range(0, rows, _REFERENCE_ROW_TILE):
        row_end = min(row_start + _REFERENCE_ROW_TILE, rows)
        positions = torch.arange(
            row_start * columns,
            row_end * columns,
            device=packed.device,
            dtype=torch.long,
        )
        values = packed[positions // 4]
        shifts = (positions.remainder(4) * 2).to(torch.uint8)
        ternary = ((values >> shifts) & 0x03).to(torch.int32).sub_(1).view(row_end - row_start, columns)
        if flat_inputs.device.type == "cuda":
            # CUDA does not expose an arbitrary-shape INT32 GEMM.  Every
            # operand and partial sum here is an exactly representable integer
            # in FP32 for BitVLA's linear dimensions, so FP32 is an exact
            # portable oracle for this integer dot product.
            accumulated = flat_inputs.float().matmul(ternary.T.float())
        else:
            accumulated = flat_inputs.to(torch.int32).matmul(ternary.T).to(torch.float32)
        row_scale = (
            flat_weight_scale[:1]
            if weight_scale.numel() == 1
            else flat_weight_scale[row_start:row_end]
        )
        output[:, row_start:row_end] = accumulated * flat_input_scale * row_scale

    if bias is not None:
        output.add_(bias.reshape(1, rows).to(dtype=output.dtype))
    return output.to(output_dtype).view(*input_shape[:-1], rows)


def direct_packed_int8_linear_reference(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Quantize exactly as BitVLA, then run the portable direct packed reference."""
    _validate_packed_projection(inputs, packed, weight_scale, bias, rows, columns)
    quantized, activation_scale = bitvla_quantize_activation_int8(inputs)
    input_inverse_scale = activation_scale.reciprocal()
    return packed_ternary_int8_linear_reference(
        quantized,
        input_inverse_scale,
        packed,
        weight_scale,
        bias,
        rows,
        columns,
        output_dtype=inputs.dtype,
    )


if triton is not None:

    @triton.jit
    def _bitvla_rowwise_int8_quant_kernel(
        inputs,
        quantized,
        scale,
        M,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One upstream-ActQuant token per program, with no intermediate BF16 tensor."""
        row = tl.program_id(axis=0)
        offsets_k = tl.arange(0, BLOCK_K)
        mask = (row < M) & (offsets_k < K)
        values = tl.load(inputs + row * K + offsets_k, mask=mask, other=0.0).to(tl.float32)
        # absmax is associative; this reduction has the same FP32 result as
        # torch.abs(...).amax for finite BitVLA activations.
        absmax = tl.max(tl.abs(values), axis=0)
        token_scale = libdevice.div_rn(127.0, tl.maximum(absmax, 1e-5))
        # ``rint`` uses IEEE round-to-nearest-even, matching torch.round.
        rounded = libdevice.rint(libdevice.mul_rn(values, token_scale))
        codes = tl.maximum(tl.minimum(rounded, 127.0), -128.0)
        tl.store(quantized + row * K + offsets_k, codes.to(tl.int8), mask=mask)
        tl.store(scale + row, token_scale, mask=row < M)


    @triton.jit
    def _bitvla_int8_quant_from_scale_kernel(
        inputs,
        scale,
        quantized,
        M,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Apply an already-exact upstream scale in one quantize/store kernel."""
        row = tl.program_id(axis=0)
        offsets_k = tl.arange(0, BLOCK_K)
        mask = (row < M) & (offsets_k < K)
        values = tl.load(inputs + row * K + offsets_k, mask=mask, other=0.0).to(tl.float32)
        token_scale = tl.load(scale + row, mask=row < M, other=0.0)
        codes = tl.maximum(tl.minimum(libdevice.rint(values * token_scale), 127.0), -128.0)
        tl.store(quantized + row * K + offsets_k, codes.to(tl.int8), mask=mask)

    @triton.jit
    def _packed_ternary_int8_gemm_kernel(
        quantized_inputs,
        input_inverse_scale,
        packed_weight,
        weight_scale,
        bias,
        output,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        PER_CHANNEL_SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """INT8 GEMM with register-resident ternary decoding from packed bytes."""
        program_m = tl.program_id(axis=0)
        program_n = tl.program_id(axis=1)
        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulated = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)

        for start_k in range(0, K, BLOCK_K):
            input_mask = (offsets_m[:, None] < M) & (start_k + offsets_k[None, :] < K)
            input_tile = tl.load(
                quantized_inputs + offsets_m[:, None] * K + start_k + offsets_k[None, :],
                mask=input_mask,
                other=0,
            )
            # BitVLA packs flattened row-major codes.  This direct load/decode
            # path is the key distinction from the former full-matrix unpack.
            positions = offsets_n[:, None] * K + start_k + offsets_k[None, :]
            weight_mask = (offsets_n[:, None] < N) & (start_k + offsets_k[None, :] < K)
            packed_values = tl.load(packed_weight + positions // 4, mask=weight_mask, other=1)
            shifts = (positions % 4) * 2
            ternary_tile = ((packed_values >> shifts) & 0x03).to(tl.int8) - 1
            accumulated = tl.dot(
                input_tile,
                tl.trans(ternary_tile),
                accumulated,
                out_dtype=tl.int32,
            )

        input_step = tl.load(input_inverse_scale + offsets_m, mask=offsets_m < M, other=0.0)
        if PER_CHANNEL_SCALE:
            output_step = tl.load(weight_scale + offsets_n, mask=offsets_n < N, other=0.0)
        else:
            output_step = tl.load(weight_scale)
        values = accumulated.to(tl.float32) * input_step[:, None] * output_step[None, :]
        if HAS_BIAS:
            values += tl.load(bias + offsets_n, mask=offsets_n < N, other=0.0)[None, :]
        output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
        tl.store(output + offsets_m[:, None] * N + offsets_n[None, :], values, mask=output_mask)


    @triton.jit
    def _packed_ternary_int8_lane4_gemm_kernel(
        quantized_inputs,
        input_inverse_scale,
        packed_weight,
        weight_scale,
        bias,
        output,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        PER_CHANNEL_SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Direct packed GEMM that loads each packed byte once for four dot tiles."""
        program_m = tl.program_id(axis=0)
        program_n = tl.program_id(axis=1)
        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k4 = tl.arange(0, BLOCK_K // 4)
        accumulated = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)

        for start_k in range(0, K, BLOCK_K):
            weight_mask = (offsets_n[:, None] < N) & (start_k + 4 * offsets_k4[None, :] < K)
            byte_positions = offsets_n[:, None] * K + start_k + 4 * offsets_k4[None, :]
            # The load is shared by all four adjacent two-bit codes.  This is
            # the packed-weight traffic/decode reduction absent from the
            # scalar direct kernel above.
            packed_values = tl.load(packed_weight + byte_positions // 4, mask=weight_mask, other=1)
            for lane in range(4):
                offsets_k = start_k + 4 * offsets_k4 + lane
                input_mask = (offsets_m[:, None] < M) & (offsets_k[None, :] < K)
                input_tile = tl.load(
                    quantized_inputs + offsets_m[:, None] * K + offsets_k[None, :],
                    mask=input_mask,
                    other=0,
                )
                ternary_tile = ((packed_values >> (lane * 2)) & 0x03).to(tl.int8) - 1
                accumulated = tl.dot(
                    input_tile,
                    tl.trans(ternary_tile),
                    accumulated,
                    out_dtype=tl.int32,
                )

        input_step = tl.load(input_inverse_scale + offsets_m, mask=offsets_m < M, other=0.0)
        if PER_CHANNEL_SCALE:
            output_step = tl.load(weight_scale + offsets_n, mask=offsets_n < N, other=0.0)
        else:
            output_step = tl.load(weight_scale)
        values = accumulated.to(tl.float32) * input_step[:, None] * output_step[None, :]
        if HAS_BIAS:
            values += tl.load(bias + offsets_n, mask=offsets_n < N, other=0.0)[None, :]
        output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
        tl.store(output + offsets_m[:, None] * N + offsets_n[None, :], values, mask=output_mask)


    @triton.jit
    def _packed_ternary_bf16_gemm_kernel(
        quantized_inputs,
        activation_scale,
        packed_weight,
        weight_scale,
        bias,
        output,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        PER_CHANNEL_SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Direct packed kernel that reconstructs upstream BF16 operands in tiles."""
        program_m = tl.program_id(axis=0)
        program_n = tl.program_id(axis=1)
        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulated = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        input_step = tl.load(activation_scale + offsets_m, mask=offsets_m < M, other=1.0)
        if PER_CHANNEL_SCALE:
            output_step = tl.load(weight_scale + offsets_n, mask=offsets_n < N, other=0.0)
        else:
            output_step = tl.load(weight_scale)

        for start_k in range(0, K, BLOCK_K):
            input_mask = (offsets_m[:, None] < M) & (start_k + offsets_k[None, :] < K)
            input_codes = tl.load(
                quantized_inputs + offsets_m[:, None] * K + start_k + offsets_k[None, :],
                mask=input_mask,
                other=0,
            )
            # ``q / scale`` and the BF16 cast duplicate upstream ActQuant.
            input_tile = (input_codes.to(tl.float32) / input_step[:, None]).to(tl.bfloat16)
            positions = offsets_n[:, None] * K + start_k + offsets_k[None, :]
            weight_mask = (offsets_n[:, None] < N) & (start_k + offsets_k[None, :] < K)
            packed_values = tl.load(packed_weight + positions // 4, mask=weight_mask, other=1)
            shifts = (positions % 4) * 2
            ternary_tile = ((packed_values >> shifts) & 0x03).to(tl.float32) - 1.0
            weight_tile = (ternary_tile * output_step[:, None]).to(tl.bfloat16)
            accumulated = tl.dot(
                input_tile,
                tl.trans(weight_tile),
                accumulated,
                input_precision="ieee",
                out_dtype=tl.float32,
            )

        values = accumulated
        if HAS_BIAS:
            values += tl.load(bias + offsets_n, mask=offsets_n < N, other=0.0)[None, :]
        output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
        tl.store(output + offsets_m[:, None] * N + offsets_n[None, :], values, mask=output_mask)


def triton_bitvla_quantize_activation_int8(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental all-Triton BitVLA activation prepass.

    This avoids redundant reduction work across output-N programs but can
    differ from ATen by one FP32 division ULP.  It must not be used for an
    exact-action evaluation gate; use ``hybrid_bitvla_quantize_activation_int8``
    or the default PyTorch activation backend instead.
    """
    if not triton_packed_kernel_available():
        raise RuntimeError("Triton activation quantization requires an available CUDA device")
    if inputs.device.type != "cuda":
        raise ValueError("Triton activation quantization requires CUDA tensors")
    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    columns = inputs.shape[-1]
    block_k = triton.next_power_of_2(columns)
    # BitVLA's text/vision projections are at most 6912-wide.  Keep an exact
    # eager fallback instead of silently compiling an impractical giant tile
    # for an unforeseen wider model.
    if block_k > 8192:
        return bitvla_quantize_activation_int8(inputs)
    flat_inputs = inputs.reshape(-1, columns).contiguous()
    quantized = torch.empty_like(flat_inputs, dtype=torch.int8)
    scale = torch.empty((flat_inputs.shape[0], 1), device=inputs.device, dtype=torch.float32)
    _bitvla_rowwise_int8_quant_kernel[(flat_inputs.shape[0],)](
        flat_inputs,
        quantized,
        scale,
        flat_inputs.shape[0],
        K=columns,
        BLOCK_K=block_k,
        num_warps=4 if block_k <= 1024 else 8,
    )
    return quantized.view_as(inputs), scale.view(*inputs.shape[:-1], 1)


def hybrid_bitvla_quantize_activation_int8(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use upstream-exact PyTorch scale reduction plus fused Triton code generation.

    The fully fused prepass above can differ by one FP32 division ULP from
    ATen's scale.  This hybrid path keeps that decisive scale operation in
    upstream-compatible PyTorch while removing the subsequent eager
    multiply/round/clamp/cast chain.  It is usable only if its exact-code test
    passes for the deployed GPU/runtime.
    """
    if not triton_packed_kernel_available():
        raise RuntimeError("Hybrid activation quantization requires an available CUDA device")
    if inputs.device.type != "cuda":
        raise ValueError("Hybrid activation quantization requires CUDA tensors")
    columns = inputs.shape[-1]
    block_k = triton.next_power_of_2(columns)
    if block_k > 8192:
        return bitvla_quantize_activation_int8(inputs)
    values = inputs.float()
    scale = 127 / values.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    flat_inputs = inputs.reshape(-1, columns).contiguous()
    quantized = torch.empty_like(flat_inputs, dtype=torch.int8)
    _bitvla_int8_quant_from_scale_kernel[(flat_inputs.shape[0],)](
        flat_inputs,
        scale.reshape(-1).contiguous(),
        quantized,
        flat_inputs.shape[0],
        K=columns,
        BLOCK_K=block_k,
        num_warps=4 if block_k <= 1024 else 8,
    )
    return quantized.view_as(inputs), scale


def _quantize_activation_for_direct_kernel(
    inputs: torch.Tensor,
    activation_backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if activation_backend not in {"auto", "torch", "triton", "hybrid"}:
        raise ValueError(f"unsupported activation backend: {activation_backend}")
    if activation_backend == "triton":
        return triton_bitvla_quantize_activation_int8(inputs)
    if activation_backend == "hybrid":
        return hybrid_bitvla_quantize_activation_int8(inputs)
    return bitvla_quantize_activation_int8(inputs)


def _triton_packed_ternary_int8_linear(
    quantized_inputs: torch.Tensor,
    input_inverse_scale: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
    *,
    output_dtype: torch.dtype,
    kernel_config: PackedKernelConfig | None = None,
    decode_mode: str = "auto",
) -> torch.Tensor:
    """Launch the direct packed CUDA kernel after exact activation quantization."""
    if not triton_packed_kernel_available():
        raise RuntimeError("Triton direct packed kernel requires an available CUDA device")
    if quantized_inputs.device.type != "cuda":
        raise ValueError("Triton direct packed kernel requires CUDA tensors")
    if quantized_inputs.dtype != torch.int8:
        raise TypeError(f"quantized inputs must be torch.int8, got {quantized_inputs.dtype}")
    _validate_packed_projection(quantized_inputs, packed, weight_scale, bias, rows, columns)
    if decode_mode not in {"auto", "scalar", "lane4"}:
        raise ValueError(f"unsupported packed decode mode: {decode_mode}")

    input_shape = quantized_inputs.shape
    flat_inputs = quantized_inputs.reshape(-1, columns).contiguous()
    flat_scale = input_inverse_scale.reshape(-1).contiguous()
    output = torch.empty((flat_inputs.shape[0], rows), device=flat_inputs.device, dtype=output_dtype)
    # Shape specialization is intentional: larger prompt/image prefill tiles
    # reduce repeated packed-weight decode work without imposing their register
    # footprint on the short action-token path.
    config = kernel_config or _select_packed_int8_config(flat_inputs.shape[0], rows, columns)
    config.validate()
    use_lane4 = decode_mode == "lane4" or (
        decode_mode == "auto" and columns % 4 == 0 and config.block_k >= 128 and config.block_k % 4 == 0
    )
    if use_lane4 and (columns % 4 or config.block_k < 128 or config.block_k % 4):
        raise ValueError("lane4 decoding requires columns and block_k divisible by at least 128")
    grid = (triton.cdiv(flat_inputs.shape[0], config.block_m), triton.cdiv(rows, config.block_n))
    bias_pointer = output if bias is None else bias.contiguous()
    kernel = _packed_ternary_int8_lane4_gemm_kernel if use_lane4 else _packed_ternary_int8_gemm_kernel
    kernel[grid](
        flat_inputs,
        flat_scale,
        packed.contiguous(),
        weight_scale.reshape(-1).contiguous(),
        bias_pointer,
        output,
        M=flat_inputs.shape[0],
        N=rows,
        K=columns,
        HAS_BIAS=bias is not None,
        PER_CHANNEL_SCALE=weight_scale.numel() != 1,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,
    )
    return output.view(*input_shape[:-1], rows)


def packed_ternary_bf16_linear_reference(
    quantized_inputs: torch.Tensor,
    activation_scale: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Tiled upstream-BF16 oracle with no full decoded weight allocation."""
    _validate_packed_projection(
        quantized_inputs,
        packed,
        weight_scale,
        bias,
        rows,
        columns,
    )
    if quantized_inputs.dtype != torch.int8:
        raise TypeError(f"quantized inputs must be torch.int8, got {quantized_inputs.dtype}")
    if activation_scale.shape != (*quantized_inputs.shape[:-1], 1):
        raise ValueError("activation_scale must have shape inputs.shape[:-1] + (1,)")

    input_shape = quantized_inputs.shape
    flat_inputs = quantized_inputs.reshape(-1, columns)
    flat_scale = activation_scale.reshape(-1, 1)
    # The conversion is deliberately before every projection, as in ActQuant.
    activations = (flat_inputs.float() / flat_scale).to(output_dtype)
    output = torch.empty((flat_inputs.shape[0], rows), device=flat_inputs.device, dtype=output_dtype)
    flat_weight_scale = _weight_scale_rows(weight_scale, rows).to(dtype=torch.float32)
    for row_start in range(0, rows, _REFERENCE_ROW_TILE):
        row_end = min(row_start + _REFERENCE_ROW_TILE, rows)
        positions = torch.arange(
            row_start * columns,
            row_end * columns,
            device=packed.device,
            dtype=torch.long,
        )
        values = packed[positions // 4]
        shifts = (positions.remainder(4) * 2).to(torch.uint8)
        ternary = ((values >> shifts) & 0x03).to(torch.float32).sub_(1).view(row_end - row_start, columns)
        row_scale = (
            flat_weight_scale[:1]
            if weight_scale.numel() == 1
            else flat_weight_scale[row_start:row_end]
        )
        weights = (ternary * row_scale[:, None]).to(output_dtype)
        tile_bias = None if bias is None else bias[row_start:row_end]
        output[:, row_start:row_end] = torch.nn.functional.linear(activations, weights, tile_bias)
    return output.view(*input_shape[:-1], rows)


def _triton_packed_ternary_bf16_linear(
    quantized_inputs: torch.Tensor,
    activation_scale: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Launch the direct packed BF16 candidate without dense weight decode."""
    if not triton_packed_kernel_available():
        raise RuntimeError("Triton direct packed kernel requires an available CUDA device")
    if quantized_inputs.device.type != "cuda":
        raise ValueError("Triton direct packed kernel requires CUDA tensors")
    if quantized_inputs.dtype != torch.int8:
        raise TypeError(f"quantized inputs must be torch.int8, got {quantized_inputs.dtype}")
    _validate_packed_projection(quantized_inputs, packed, weight_scale, bias, rows, columns)

    input_shape = quantized_inputs.shape
    flat_inputs = quantized_inputs.reshape(-1, columns).contiguous()
    flat_scale = activation_scale.reshape(-1).contiguous()
    output = torch.empty((flat_inputs.shape[0], rows), device=flat_inputs.device, dtype=output_dtype)
    block_m = 16 if flat_inputs.shape[0] >= 16 else 8
    block_n = 64 if rows >= 64 else 32
    block_k = 32
    grid = (triton.cdiv(flat_inputs.shape[0], block_m), triton.cdiv(rows, block_n))
    bias_pointer = output if bias is None else bias.contiguous()
    _packed_ternary_bf16_gemm_kernel[grid](
        flat_inputs,
        flat_scale,
        packed.contiguous(),
        weight_scale.reshape(-1).contiguous(),
        bias_pointer,
        output,
        M=flat_inputs.shape[0],
        N=rows,
        K=columns,
        HAS_BIAS=bias is not None,
        PER_CHANNEL_SCALE=weight_scale.numel() != 1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output.view(*input_shape[:-1], rows)


def direct_packed_bf16_linear(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
    *,
    backend: str = "auto",
    activation_backend: str = "torch",
) -> torch.Tensor:
    """Experimental direct packed BF16 candidate for the action-parity gate.

    It recreates upstream BF16 activation and weight values within a Triton
    tile, then uses FP32 accumulation.  It is *not* eligible for model runtime
    use until an end-to-end BitVLA action comparison proves exact parity.  Its
    default ``activation_backend='torch'`` deliberately favors that fidelity.
    """
    _validate_packed_projection(inputs, packed, weight_scale, bias, rows, columns)
    if backend not in {"auto", "triton", "reference"}:
        raise ValueError(f"unsupported packed direct backend: {backend}")
    quantized, activation_scale = _quantize_activation_for_direct_kernel(inputs, activation_backend)
    use_triton = backend == "triton" or (
        backend == "auto" and inputs.device.type == "cuda" and triton_packed_kernel_available()
    )
    if not use_triton:
        return packed_ternary_bf16_linear_reference(
            quantized,
            activation_scale,
            packed,
            weight_scale,
            bias,
            rows,
            columns,
            output_dtype=inputs.dtype,
        )
    return _triton_packed_ternary_bf16_linear(
        quantized,
        activation_scale,
        packed,
        weight_scale,
        bias,
        rows,
        columns,
        output_dtype=inputs.dtype,
    )


def direct_packed_int8_linear(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
    *,
    backend: str = "auto",
    kernel_config: PackedKernelConfig | None = None,
    activation_backend: str = "torch",
    decode_mode: str = "auto",
) -> torch.Tensor:
    """Project with packed ternary weights and exact BitVLA INT8 activations.

    ``backend='triton'`` launches the direct packed CUDA kernel.  ``'reference'``
    uses the portable tiled oracle; ``'auto'`` selects Triton only when it is
    actually available.  ``kernel_config`` permits reproducible shape tuning;
    ``decode_mode='lane4'`` loads each packed byte once for its four codes.
    Neither backend materializes the complete decoded BF16 weight matrix.  The
    exact default uses upstream PyTorch activation quantization; pass
    ``activation_backend='hybrid'`` only after its exact-code GPU check.
    """
    _validate_packed_projection(inputs, packed, weight_scale, bias, rows, columns)
    if backend not in {"auto", "triton", "reference"}:
        raise ValueError(f"unsupported packed direct backend: {backend}")
    if decode_mode not in {"auto", "scalar", "lane4"}:
        raise ValueError(f"unsupported packed decode mode: {decode_mode}")
    if kernel_config is not None:
        kernel_config.validate()
    use_triton = backend == "triton" or (
        backend == "auto" and inputs.device.type == "cuda" and triton_packed_kernel_available()
    )
    if not use_triton:
        return direct_packed_int8_linear_reference(
            inputs,
            packed,
            weight_scale,
            bias,
            rows,
            columns,
        )

    quantized, activation_scale = _quantize_activation_for_direct_kernel(inputs, activation_backend)
    input_inverse_scale = activation_scale.reciprocal()
    return _triton_packed_ternary_int8_linear(
        quantized,
        input_inverse_scale,
        packed,
        weight_scale,
        bias,
        rows,
        columns,
        output_dtype=inputs.dtype,
        kernel_config=kernel_config,
        decode_mode=decode_mode,
    )
