"""Direct packed-ternary INT8 projection kernels for native BitVLA.

The kernels deliberately never create a dense BF16 or INT8 weight
matrix.  They read BitVLA's four two-bit codes per byte from global memory,
decode a tile to ternary values in registers, and multiply that tile by
already-quantized INT8 activations.  Activations reproduce upstream
``ActQuant`` exactly: one FP32 absolute-max scale per input token,
round-to-nearest, saturation to the signed INT8 range, followed by BF16
dequantization when the BF16 exact-candidate path is selected.  The default
activation backend is the exact upstream PyTorch expression; the all-Triton
activation prepass is retained only as an explicitly approximate experiment.

The result is mathematically equivalent to the integer projection
``(q_x @ ternary_weight.T) * input_step * weight_step``.  Two CUDA mainloops
share that contract:

* a Tensor-Core GEMM for prompt/image batches; and
* a DP4A GEMV path for the latency-sensitive one/few-token case.

The DP4A path follows the interleaving principle published with Microsoft's
MIT-licensed BitNet GPU kernel, but implements BitWAM's shape-generic launch,
FP32 epilogue, optional bias, and PyTorch/Triton integration here.  Its packed
layout is still exactly two bits per ternary weight; it is only a 4-by-4 bit
transpose within each group of sixteen weights, not an unpacked cache.
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
_DP4A_WEIGHTS_PER_WORD: Final = 16
_DP4A_BYTES_PER_WORD: Final = 4


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


@dataclass(frozen=True)
class PackedDp4aConfig:
    """A shape-specialized direct packed DP4A launch configuration."""

    block_n: int = 16
    block_groups: int = 32
    num_warps: int = 4

    def validate(self) -> None:
        for name, value in (
            ("block_n", self.block_n),
            ("block_groups", self.block_groups),
            ("num_warps", self.num_warps),
        ):
            if value < 1 or value & (value - 1):
                raise ValueError(f"{name} must be a positive power of two")
        if self.block_n > 64:
            raise ValueError("block_n must not exceed 64 for the DP4A reduction")


def repack_packed_ternary_for_dp4a(
    packed: torch.Tensor,
    rows: int,
    columns: int,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Transpose packed 2-bit codes into the DP4A decode layout.

    The operation is an involution: applying it a second time restores
    BitVLA's flattened row-major packing.  It maps four source bytes holding
    ``[w0..w3], [w4..w7], [w8..w11], [w12..w15]`` to four bytes holding
    ``[w0,w4,w8,w12], ...``.  A 32-bit load can then expose four consecutive
    ternary values with one shift and one mask, ready for ``dp4a``.

    No dense weight is created: input and output both contain exactly one
    two-bit code per weight.
    """
    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    if columns % _DP4A_WEIGHTS_PER_WORD:
        raise ValueError("DP4A packing requires columns divisible by 16")
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed weight must be torch.uint8, got {packed.dtype}")
    if packed.numel() != rows * columns // 4:
        raise ValueError("packed weight does not contain exactly four two-bit codes per byte")
    if backend not in {"auto", "triton", "torch"}:
        raise ValueError(f"unsupported packed repack backend: {backend}")
    use_triton = backend == "triton" or (
        backend == "auto" and packed.device.type == "cuda" and triton_packed_kernel_available()
    )
    if use_triton:
        if packed.device.type != "cuda" or not triton_packed_kernel_available():
            raise RuntimeError("Triton packed repacking requires an available CUDA device")
        output = torch.empty_like(packed)
        block = 256
        _repack_packed_ternary_dp4a_kernel[(triton.cdiv(packed.numel(), block),)](
            packed.contiguous(),
            output,
            packed.numel(),
            BLOCK=block,
            num_warps=4,
        )
        return output

    source = packed.reshape(rows, columns // _DP4A_WEIGHTS_PER_WORD, _DP4A_BYTES_PER_WORD)
    source_words = source.to(torch.int32)
    target_bytes = []
    for source_lane in range(_DP4A_BYTES_PER_WORD):
        codes = (source_words >> (2 * source_lane)) & 0x03
        target_bytes.append(
            codes[..., 0]
            | (codes[..., 1] << 2)
            | (codes[..., 2] << 4)
            | (codes[..., 3] << 6)
        )
    return torch.stack(target_bytes, dim=-1).to(torch.uint8).reshape(-1).contiguous()


def _select_packed_int8_config(tokens: int, rows: int, columns: int) -> PackedKernelConfig:
    """Favor output/sequence tiles that reuse direct-decoded packed weights."""
    if tokens >= 128 and rows >= 128 and columns >= 64:
        # Packed decode is amortized over the largest M tile that does not
        # force excessive accumulator spilling.  Wide output projections favor
        # a narrower N tile; wide reduction/down projections favor a larger K
        # tile.  These choices are the reproducible A100/B200 tuning family,
        # not shape-specific hard-coded model dimensions.
        if rows >= 4096:
            return PackedKernelConfig(block_m=256, block_n=64, block_k=64, num_warps=4)
        if columns >= 4096 and rows <= 2048:
            return PackedKernelConfig(block_m=128, block_n=64, block_k=128, num_warps=8)
        return PackedKernelConfig(block_m=128, block_n=128, block_k=128, num_warps=8)
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
    def _repack_packed_ternary_dp4a_kernel(source, target, packed_elements, BLOCK: tl.constexpr):
        """Transpose each four-byte 2-bit block without an unpacked intermediate."""
        target_index = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
        mask = target_index < packed_elements
        group_base = (target_index // 4) * 4
        source_lane = target_index % 4
        source_0 = tl.load(source + group_base, mask=mask, other=0).to(tl.uint32)
        source_1 = tl.load(source + group_base + 1, mask=mask, other=0).to(tl.uint32)
        source_2 = tl.load(source + group_base + 2, mask=mask, other=0).to(tl.uint32)
        source_3 = tl.load(source + group_base + 3, mask=mask, other=0).to(tl.uint32)
        shift = source_lane * 2
        value = (
            ((source_0 >> shift) & 0x03)
            | (((source_1 >> shift) & 0x03) << 2)
            | (((source_2 >> shift) & 0x03) << 4)
            | (((source_3 >> shift) & 0x03) << 6)
        )
        tl.store(target + target_index, value.to(tl.uint8), mask=mask)

    @triton.jit
    def _bitvla_rowwise_int8_quant_kernel(
        inputs,
        quantized,
        scale,
        row_sum,
        M,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        STORE_ROW_SUM: tl.constexpr,
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
        if STORE_ROW_SUM:
            tl.store(row_sum + row, tl.sum(codes.to(tl.int32), axis=0), mask=row < M)


    @triton.jit
    def _bitvla_int8_quant_from_scale_kernel(
        inputs,
        scale,
        quantized,
        row_sum,
        M,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        STORE_ROW_SUM: tl.constexpr,
    ):
        """Apply an already-exact upstream scale in one quantize/store kernel."""
        row = tl.program_id(axis=0)
        offsets_k = tl.arange(0, BLOCK_K)
        mask = (row < M) & (offsets_k < K)
        values = tl.load(inputs + row * K + offsets_k, mask=mask, other=0.0).to(tl.float32)
        token_scale = tl.load(scale + row, mask=row < M, other=0.0)
        codes = tl.maximum(tl.minimum(libdevice.rint(values * token_scale), 127.0), -128.0)
        tl.store(quantized + row * K + offsets_k, codes.to(tl.int8), mask=mask)
        if STORE_ROW_SUM:
            tl.store(row_sum + row, tl.sum(codes.to(tl.int32), axis=0), mask=row < M)


    @triton.jit
    def _dp4a_s32(lhs, rhs, accumulated):
        """Four signed INT8 products accumulated exactly into one INT32."""
        return tl.inline_asm_elementwise(
            asm="dp4a.s32.s32 $0, $1, $2, $3;",
            constraints="=r,r,r,r",
            args=[lhs, rhs, accumulated],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _packed_ternary_dp4a_gemv_kernel(
        quantized_inputs_i32,
        activation_scale,
        activation_sum,
        packed_weight_i32,
        weight_scale,
        bias,
        output,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        PER_CHANNEL_SCALE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_GROUPS: tl.constexpr,
    ):
        """Direct W2A8 GEMV from interleaved two-bit weights.

        Each packed 32-bit word represents sixteen ternary codes.  A shift and
        byte mask expose four consecutive non-negative codes for ``dp4a``.
        Since the stored code is ``weight + 1``, subtracting the activation-row
        sum once after the reduction recovers the signed ternary dot product.
        """
        row = tl.program_id(axis=0)
        program_n = tl.program_id(axis=1)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_group = tl.arange(0, BLOCK_GROUPS)
        groups = K // 16
        accumulated = tl.zeros((BLOCK_N,), tl.int32)

        for start_group in range(0, groups, BLOCK_GROUPS):
            group = start_group + offsets_group
            group_mask = group < groups
            weight_mask = (offsets_n[:, None] < N) & group_mask[None, :]
            packed_words = tl.load(
                packed_weight_i32 + offsets_n[:, None] * groups + group[None, :],
                mask=weight_mask,
                other=0,
            ).to(tl.uint32)
            partial = tl.zeros((BLOCK_N, BLOCK_GROUPS), tl.int32)
            for lane in range(4):
                # The packed 4-by-4 transpose makes each decoded byte lane
                # correspond to one contiguous group of four activations.
                decoded = ((packed_words >> (2 * lane)) & 0x03030303).to(tl.int32)
                activation_words = tl.load(
                    quantized_inputs_i32 + row * (K // 4) + group * 4 + lane,
                    mask=(row < M) & group_mask,
                    other=0,
                )
                partial = _dp4a_s32(activation_words[None, :], decoded, partial)
            accumulated += tl.sum(partial, axis=1)

        # code = ternary + 1, therefore dot(x, ternary) = dot(x, code) - sum(x).
        accumulated -= tl.load(activation_sum + row, mask=row < M, other=0)
        token_scale = tl.load(activation_scale + row, mask=row < M, other=1.0)
        if PER_CHANNEL_SCALE:
            output_step = tl.load(weight_scale + offsets_n, mask=offsets_n < N, other=0.0)
        else:
            output_step = tl.load(weight_scale)
        input_step = libdevice.div_rn(1.0, token_scale)
        values = accumulated.to(tl.float32) * input_step * output_step
        if HAS_BIAS:
            values += tl.load(bias + offsets_n, mask=offsets_n < N, other=0.0)
        tl.store(output + row * N + offsets_n, values, mask=(row < M) & (offsets_n < N))

    @triton.jit
    def _packed_ternary_int8_gemm_kernel(
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
        DP4A_LAYOUT: tl.constexpr,
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
            logical_k = start_k + offsets_k[None, :]
            weight_mask = (offsets_n[:, None] < N) & (start_k + offsets_k[None, :] < K)
            if DP4A_LAYOUT:
                byte_in_row = (logical_k // 16) * 4 + logical_k % 4
                packed_positions = offsets_n[:, None] * (K // 4) + byte_in_row
                shifts = ((logical_k % 16) // 4) * 2
            else:
                positions = offsets_n[:, None] * K + logical_k
                packed_positions = positions // 4
                shifts = (positions % 4) * 2
            packed_values = tl.load(packed_weight + packed_positions, mask=weight_mask, other=1)
            ternary_tile = ((packed_values >> shifts) & 0x03).to(tl.int8) - 1
            accumulated = tl.dot(
                input_tile,
                tl.trans(ternary_tile),
                accumulated,
                out_dtype=tl.int32,
            )

        token_scale = tl.load(activation_scale + offsets_m, mask=offsets_m < M, other=1.0)
        input_step = libdevice.div_rn(1.0, token_scale)
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
        DP4A_LAYOUT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Direct packed GEMM that loads each packed byte once for four dot tiles."""
        tl.static_assert(not DP4A_LAYOUT, "lane4 requires row-major packing")
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

        token_scale = tl.load(activation_scale + offsets_m, mask=offsets_m < M, other=1.0)
        input_step = libdevice.div_rn(1.0, token_scale)
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


def _triton_bitvla_quantize_activation_int8(
    inputs: torch.Tensor,
    *,
    return_row_sum: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the all-Triton prepass, optionally producing the DP4A correction sum.

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
        quantized, scale = bitvla_quantize_activation_int8(inputs)
        row_sum = quantized.to(torch.int32).sum(dim=-1) if return_row_sum else None
        return quantized, scale, row_sum
    flat_inputs = inputs.reshape(-1, columns).contiguous()
    quantized = torch.empty_like(flat_inputs, dtype=torch.int8)
    scale = torch.empty((flat_inputs.shape[0], 1), device=inputs.device, dtype=torch.float32)
    row_sum = (
        torch.empty((flat_inputs.shape[0],), device=inputs.device, dtype=torch.int32)
        if return_row_sum
        else None
    )
    _bitvla_rowwise_int8_quant_kernel[(flat_inputs.shape[0],)](
        flat_inputs,
        quantized,
        scale,
        quantized if row_sum is None else row_sum,
        flat_inputs.shape[0],
        K=columns,
        BLOCK_K=block_k,
        STORE_ROW_SUM=return_row_sum,
        num_warps=4 if block_k <= 1024 else 8,
    )
    shaped_sum = None if row_sum is None else row_sum.view(*inputs.shape[:-1])
    return quantized.view_as(inputs), scale.view(*inputs.shape[:-1], 1), shaped_sum


def triton_bitvla_quantize_activation_int8(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental all-Triton BitVLA activation prepass.

    This avoids redundant reduction work across output-N programs but can
    differ from ATen by one FP32 division ULP.  It must not be used for an
    exact-action evaluation gate; use ``hybrid_bitvla_quantize_activation_int8``
    or the default PyTorch activation backend instead.
    """
    quantized, scale, _ = _triton_bitvla_quantize_activation_int8(
        inputs,
        return_row_sum=False,
    )
    return quantized, scale


def _hybrid_bitvla_quantize_activation_int8(
    inputs: torch.Tensor,
    *,
    return_row_sum: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run exact-scale hybrid quantization, optionally producing a row sum.

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
        quantized, scale = bitvla_quantize_activation_int8(inputs)
        row_sum = quantized.to(torch.int32).sum(dim=-1) if return_row_sum else None
        return quantized, scale, row_sum
    values = inputs.float()
    scale = 127 / values.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    flat_inputs = inputs.reshape(-1, columns).contiguous()
    quantized = torch.empty_like(flat_inputs, dtype=torch.int8)
    row_sum = (
        torch.empty((flat_inputs.shape[0],), device=inputs.device, dtype=torch.int32)
        if return_row_sum
        else None
    )
    _bitvla_int8_quant_from_scale_kernel[(flat_inputs.shape[0],)](
        flat_inputs,
        scale.reshape(-1).contiguous(),
        quantized,
        quantized if row_sum is None else row_sum,
        flat_inputs.shape[0],
        K=columns,
        BLOCK_K=block_k,
        STORE_ROW_SUM=return_row_sum,
        num_warps=4 if block_k <= 1024 else 8,
    )
    shaped_sum = None if row_sum is None else row_sum.view(*inputs.shape[:-1])
    return quantized.view_as(inputs), scale, shaped_sum


def hybrid_bitvla_quantize_activation_int8(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use upstream-exact PyTorch scale reduction plus fused Triton code generation.

    The fully fused prepass above can differ by one FP32 division ULP from
    ATen's scale.  This hybrid path keeps that decisive scale operation in
    upstream-compatible PyTorch while removing the subsequent eager
    multiply/round/clamp/cast chain.  It is usable only if its exact-code test
    passes for the deployed GPU/runtime.
    """
    quantized, scale, _ = _hybrid_bitvla_quantize_activation_int8(
        inputs,
        return_row_sum=False,
    )
    return quantized, scale


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


def _quantize_activation_with_sum_for_direct_kernel(
    inputs: torch.Tensor,
    activation_backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize once and return the row sum required by code-minus-one DP4A."""
    if activation_backend not in {"auto", "torch", "triton", "hybrid"}:
        raise ValueError(f"unsupported activation backend: {activation_backend}")
    if activation_backend == "triton":
        quantized, scale, row_sum = _triton_bitvla_quantize_activation_int8(
            inputs,
            return_row_sum=True,
        )
    elif activation_backend == "hybrid":
        quantized, scale, row_sum = _hybrid_bitvla_quantize_activation_int8(
            inputs,
            return_row_sum=True,
        )
    else:
        quantized, scale = bitvla_quantize_activation_int8(inputs)
        row_sum = quantized.to(torch.int32).sum(dim=-1)
    if row_sum is None:  # Defensive: both sum-producing branches guarantee it.
        raise RuntimeError("activation quantization did not produce its INT8 row sum")
    return quantized, scale, row_sum


def _triton_packed_ternary_dp4a_linear(
    quantized_inputs: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_sum: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
    *,
    output_dtype: torch.dtype,
    kernel_config: PackedDp4aConfig | None = None,
) -> torch.Tensor:
    """Launch shape-generic DP4A directly over interleaved two-bit weights."""
    if not triton_packed_kernel_available():
        raise RuntimeError("Triton direct packed DP4A kernel requires an available CUDA device")
    if quantized_inputs.device.type != "cuda":
        raise ValueError("Triton direct packed DP4A kernel requires CUDA tensors")
    if quantized_inputs.dtype != torch.int8:
        raise TypeError(f"quantized inputs must be torch.int8, got {quantized_inputs.dtype}")
    if columns % _DP4A_WEIGHTS_PER_WORD:
        raise ValueError("DP4A projection requires columns divisible by 16")
    _validate_packed_projection(quantized_inputs, packed, weight_scale, bias, rows, columns)
    if activation_scale.shape != (*quantized_inputs.shape[:-1], 1):
        raise ValueError("activation_scale must have shape inputs.shape[:-1] + (1,)")
    if activation_sum.shape != quantized_inputs.shape[:-1]:
        raise ValueError("activation_sum must have shape inputs.shape[:-1]")
    if activation_sum.dtype != torch.int32:
        raise TypeError(f"activation_sum must be torch.int32, got {activation_sum.dtype}")

    input_shape = quantized_inputs.shape
    flat_inputs = quantized_inputs.reshape(-1, columns).contiguous()
    flat_scale = activation_scale.reshape(-1).contiguous()
    flat_sum = activation_sum.reshape(-1).contiguous()
    packed_contiguous = packed.contiguous()
    output = torch.empty((flat_inputs.shape[0], rows), device=flat_inputs.device, dtype=output_dtype)
    config = kernel_config or PackedDp4aConfig()
    config.validate()
    grid = (flat_inputs.shape[0], triton.cdiv(rows, config.block_n))
    bias_pointer = output if bias is None else bias.contiguous()
    _packed_ternary_dp4a_gemv_kernel[grid](
        flat_inputs.view(torch.int32),
        flat_scale,
        flat_sum,
        packed_contiguous.view(torch.int32),
        weight_scale.reshape(-1).contiguous(),
        bias_pointer,
        output,
        M=flat_inputs.shape[0],
        N=rows,
        K=columns,
        HAS_BIAS=bias is not None,
        PER_CHANNEL_SCALE=weight_scale.numel() != 1,
        BLOCK_N=config.block_n,
        BLOCK_GROUPS=config.block_groups,
        num_warps=config.num_warps,
    )
    return output.view(*input_shape[:-1], rows)


def _triton_packed_ternary_int8_linear(
    quantized_inputs: torch.Tensor,
    activation_scale: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
    *,
    output_dtype: torch.dtype,
    kernel_config: PackedKernelConfig | None = None,
    decode_mode: str = "auto",
    packed_layout: str = "row_major",
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
    if packed_layout not in {"row_major", "dp4a"}:
        raise ValueError(f"unsupported packed layout: {packed_layout}")
    if packed_layout == "dp4a" and columns % _DP4A_WEIGHTS_PER_WORD:
        raise ValueError("DP4A packing requires columns divisible by 16")
    if activation_scale.shape != (*quantized_inputs.shape[:-1], 1):
        raise ValueError("activation_scale must have shape inputs.shape[:-1] + (1,)")

    input_shape = quantized_inputs.shape
    flat_inputs = quantized_inputs.reshape(-1, columns).contiguous()
    flat_scale = activation_scale.reshape(-1).contiguous()
    output = torch.empty((flat_inputs.shape[0], rows), device=flat_inputs.device, dtype=output_dtype)
    # Shape specialization is intentional: larger prompt/image prefill tiles
    # reduce repeated packed-weight decode work without imposing their register
    # footprint on the short action-token path.
    config = kernel_config or _select_packed_int8_config(flat_inputs.shape[0], rows, columns)
    config.validate()
    use_lane4 = decode_mode == "lane4" or (
        decode_mode == "auto" and columns % 4 == 0 and config.block_k >= 128 and config.block_k % 4 == 0
    )
    if packed_layout == "dp4a":
        use_lane4 = False
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
        DP4A_LAYOUT=packed_layout == "dp4a",
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


def direct_packed_dp4a_int8_linear(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
    *,
    backend: str = "auto",
    activation_backend: str = "triton",
    kernel_config: PackedDp4aConfig | None = None,
) -> torch.Tensor:
    """Project directly from DP4A-interleaved ternary weights.

    ``packed`` contains the same number of two-bit codes as BitVLA's native
    row-major representation, transformed once with
    :func:`repack_packed_ternary_for_dp4a`.  CUDA execution loads those packed
    words directly, accumulates only integers, then applies activation scale,
    weight scale, and optional bias in the output epilogue.  It never builds a
    dense weight tensor or an INT8 weight cache.

    The portable reference converts packed-to-packed back to row-major and
    uses the tiled integer oracle.  Since the transform is an involution, that
    path also avoids a dense decoded weight.
    """
    _validate_packed_projection(inputs, packed, weight_scale, bias, rows, columns)
    if columns % _DP4A_WEIGHTS_PER_WORD:
        raise ValueError("DP4A projection requires columns divisible by 16")
    if backend not in {"auto", "triton", "reference"}:
        raise ValueError(f"unsupported packed DP4A backend: {backend}")
    if kernel_config is not None:
        kernel_config.validate()
    use_triton = backend == "triton" or (
        backend == "auto" and inputs.device.type == "cuda" and triton_packed_kernel_available()
    )
    if not use_triton:
        row_major = repack_packed_ternary_for_dp4a(packed, rows, columns)
        return direct_packed_int8_linear_reference(
            inputs,
            row_major,
            weight_scale,
            bias,
            rows,
            columns,
        )

    quantized, activation_scale, activation_sum = _quantize_activation_with_sum_for_direct_kernel(
        inputs,
        activation_backend,
    )
    return _triton_packed_ternary_dp4a_linear(
        quantized,
        activation_scale,
        activation_sum,
        packed,
        weight_scale,
        bias,
        rows,
        columns,
        output_dtype=inputs.dtype,
        kernel_config=kernel_config,
    )


def direct_packed_w2a8_linear(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int,
    columns: int,
    *,
    backend: str = "auto",
    activation_backend: str = "hybrid",
    small_batch_threshold: int = 8,
    dp4a_config: PackedDp4aConfig | None = None,
    tensorcore_config: PackedKernelConfig | None = None,
) -> torch.Tensor:
    """Dispatch one DP4A-interleaved buffer to the appropriate direct mainloop.

    One/few-token projections are memory- and launch-sensitive, so they use
    CUDA DP4A.  Larger prompt and image batches amortize packed decode across
    Tensor-Core tiles.  Both paths consume the same two-bit buffer and share
    the exact INT32-plus-FP32-epilogue numerical contract.
    """
    if small_batch_threshold < 1:
        raise ValueError("small_batch_threshold must be positive")
    tokens = inputs.numel() // columns if inputs.shape[-1] == columns else 0
    use_dp4a = tokens <= small_batch_threshold
    if use_dp4a:
        return direct_packed_dp4a_int8_linear(
            inputs,
            packed,
            weight_scale,
            bias,
            rows,
            columns,
            backend=backend,
            activation_backend=activation_backend,
            kernel_config=dp4a_config,
        )
    return direct_packed_int8_linear(
        inputs,
        packed,
        weight_scale,
        bias,
        rows,
        columns,
        backend=backend,
        kernel_config=tensorcore_config,
        activation_backend=activation_backend,
        decode_mode="scalar",
        packed_layout="dp4a",
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
    packed_layout: str = "row_major",
) -> torch.Tensor:
    """Project with packed ternary weights and exact BitVLA INT8 activations.

    ``backend='triton'`` launches the direct packed CUDA kernel.  ``'reference'``
    uses the portable tiled oracle; ``'auto'`` selects Triton only when it is
    actually available.  ``kernel_config`` permits reproducible shape tuning;
    ``decode_mode='lane4'`` loads each packed byte once for its four codes.
    ``packed_layout='dp4a'`` lets the Tensor-Core mainloop consume the same
    interleaved two-bit buffer as the DP4A GEMV path.
    Neither backend materializes the complete decoded BF16 weight matrix.  The
    exact default uses upstream PyTorch activation quantization; pass
    ``activation_backend='hybrid'`` only after its exact-code GPU check.
    """
    _validate_packed_projection(inputs, packed, weight_scale, bias, rows, columns)
    if backend not in {"auto", "triton", "reference"}:
        raise ValueError(f"unsupported packed direct backend: {backend}")
    if decode_mode not in {"auto", "scalar", "lane4"}:
        raise ValueError(f"unsupported packed decode mode: {decode_mode}")
    if packed_layout not in {"row_major", "dp4a"}:
        raise ValueError(f"unsupported packed layout: {packed_layout}")
    if packed_layout == "dp4a" and columns % _DP4A_WEIGHTS_PER_WORD:
        raise ValueError("DP4A packing requires columns divisible by 16")
    if kernel_config is not None:
        kernel_config.validate()
    use_triton = backend == "triton" or (
        backend == "auto" and inputs.device.type == "cuda" and triton_packed_kernel_available()
    )
    if not use_triton:
        reference_packed = (
            repack_packed_ternary_for_dp4a(packed, rows, columns)
            if packed_layout == "dp4a"
            else packed
        )
        return direct_packed_int8_linear_reference(
            inputs,
            reference_packed,
            weight_scale,
            bias,
            rows,
            columns,
        )

    quantized, activation_scale = _quantize_activation_for_direct_kernel(inputs, activation_backend)
    return _triton_packed_ternary_int8_linear(
        quantized,
        activation_scale,
        packed,
        weight_scale,
        bias,
        rows,
        columns,
        output_dtype=inputs.dtype,
        kernel_config=kernel_config,
        decode_mode=decode_mode,
        packed_layout=packed_layout,
    )
