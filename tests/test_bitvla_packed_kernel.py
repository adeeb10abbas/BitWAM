import pytest
import torch

from lerobot_policy_bitwam.bitvla_packed_kernel import (
    PackedDp4aConfig,
    PackedKernelConfig,
    _select_packed_int8_config,
    bitvla_quantize_activation_int8,
    direct_packed_bf16_linear,
    direct_packed_dp4a_int8_linear,
    direct_packed_int8_linear,
    direct_packed_int8_linear_reference,
    direct_packed_w2a8_linear,
    hybrid_bitvla_quantize_activation_int8,
    packed_ternary_int8_linear_reference,
    repack_packed_ternary_for_dp4a,
    triton_bitvla_quantize_activation_int8,
    triton_packed_kernel_available,
)
from lerobot_policy_bitwam.bitvla_packing import (
    dequantize_packed_weight,
    quantize_activation,
)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    flattened = codes.to(torch.int8).flatten()
    padded = torch.nn.functional.pad(flattened.add(1).to(torch.uint8), (0, (-flattened.numel()) % 4))
    values = padded.view(-1, 4)
    return values[:, 0] | values[:, 1] << 2 | values[:, 2] << 4 | values[:, 3] << 6


def _expected_direct_projection(
    inputs: torch.Tensor,
    codes: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    quantized, scale = bitvla_quantize_activation_int8(inputs)
    inverse_scale = scale.reciprocal()
    accumulated = quantized.reshape(-1, inputs.shape[-1]).to(torch.int32).matmul(codes.to(torch.int32).T)
    expected = accumulated.to(torch.float32) * inverse_scale.reshape(-1, 1)
    expected = expected * weight_scale.reshape(-1 if weight_scale.numel() > 1 else 1)
    if bias is not None:
        expected = expected + bias
    return expected.to(inputs.dtype).view(*inputs.shape[:-1], codes.shape[0])


def test_bitvla_int8_quantizer_reconstructs_upstream_fake_quantization_exactly() -> None:
    inputs = torch.tensor(
        [[0.0, -1.0, 0.5, 2.0, -0.25], [0.00390625, -0.375, 1.5, -2.0, 0.75]],
        dtype=torch.bfloat16,
    )

    codes, scale = bitvla_quantize_activation_int8(inputs)

    assert torch.equal((codes.float() / scale).to(inputs.dtype), quantize_activation(inputs))


def test_packed_kernel_config_rejects_non_tensor_core_tiles() -> None:
    with pytest.raises(ValueError, match="positive power"):
        PackedKernelConfig(block_m=12, block_n=64, block_k=32, num_warps=4).validate()
    with pytest.raises(ValueError, match="at least 16"):
        PackedKernelConfig(block_m=8, block_n=32, block_k=8, num_warps=4).validate()


def test_dp4a_kernel_config_rejects_invalid_reduction_tiles() -> None:
    with pytest.raises(ValueError, match="positive power"):
        PackedDp4aConfig(block_n=12, block_groups=32, num_warps=4).validate()
    with pytest.raises(ValueError, match="must not exceed 64"):
        PackedDp4aConfig(block_n=128, block_groups=32, num_warps=4).validate()


@pytest.mark.parametrize(
    ("tokens", "rows", "columns", "expected"),
    [
        (626, 6912, 2560, PackedKernelConfig(256, 64, 64, 4)),
        (512, 1152, 4304, PackedKernelConfig(128, 64, 128, 8)),
        (626, 2560, 2560, PackedKernelConfig(128, 128, 128, 8)),
    ],
)
def test_tensorcore_dispatch_uses_tuned_decode_amortization_family(
    tokens: int,
    rows: int,
    columns: int,
    expected: PackedKernelConfig,
) -> None:
    assert _select_packed_int8_config(tokens, rows, columns) == expected


def test_dp4a_repacking_is_an_exact_two_bit_involution() -> None:
    torch.manual_seed(7)
    rows, columns = 5, 32
    codes = torch.randint(-1, 2, (rows, columns), dtype=torch.int8)
    row_major = _pack_codes(codes)

    interleaved = repack_packed_ternary_for_dp4a(row_major, rows, columns)
    restored = repack_packed_ternary_for_dp4a(interleaved, rows, columns)

    assert interleaved.dtype == torch.uint8
    assert interleaved.numel() == codes.numel() // 4
    assert torch.equal(restored, row_major)


def test_dp4a_reference_matches_row_major_integer_projection() -> None:
    torch.manual_seed(11)
    rows, columns = 7, 32
    codes = torch.randint(-1, 2, (rows, columns), dtype=torch.int8)
    row_major = _pack_codes(codes)
    interleaved = repack_packed_ternary_for_dp4a(row_major, rows, columns)
    inputs = torch.randn(2, 3, columns, dtype=torch.bfloat16)
    scale = torch.rand(rows, dtype=torch.float32)
    bias = torch.randn(rows, dtype=torch.bfloat16)

    actual = direct_packed_dp4a_int8_linear(
        inputs,
        interleaved,
        scale,
        bias,
        rows,
        columns,
        backend="reference",
    )
    expected = direct_packed_int8_linear_reference(
        inputs,
        row_major,
        scale,
        bias,
        rows,
        columns,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("tokens", [2, 12])
def test_w2a8_dispatcher_has_one_integer_reference_across_mainloops(tokens: int) -> None:
    torch.manual_seed(13 + tokens)
    rows, columns = 7, 32
    codes = torch.randint(-1, 2, (rows, columns), dtype=torch.int8)
    row_major = _pack_codes(codes)
    interleaved = repack_packed_ternary_for_dp4a(row_major, rows, columns)
    inputs = torch.randn(tokens, columns, dtype=torch.bfloat16)
    scale = torch.tensor(0.25, dtype=torch.float32)

    actual = direct_packed_w2a8_linear(
        inputs,
        interleaved,
        scale,
        None,
        rows,
        columns,
        backend="reference",
        small_batch_threshold=8,
    )
    expected = direct_packed_int8_linear_reference(
        inputs,
        row_major,
        scale,
        None,
        rows,
        columns,
    )

    assert torch.equal(actual, expected)


def test_dp4a_repacking_rejects_non_word_aligned_columns() -> None:
    with pytest.raises(ValueError, match="divisible by 16"):
        repack_packed_ternary_for_dp4a(torch.zeros(15, dtype=torch.uint8), 3, 20)


def test_direct_reference_matches_integer_projection_with_per_channel_scales() -> None:
    codes = torch.tensor(
        [[-1, 0, 1, -1, 1], [0, 1, -1, 0, 1], [1, -1, 0, 1, -1]], dtype=torch.int8
    )
    packed = _pack_codes(codes)
    inputs = torch.tensor([[[0.0, -1.0, 0.5, 2.0, -0.25], [1.5, 0.0, -2.0, 0.25, 0.75]]])
    scale = torch.tensor([0.5, 1.25, 0.25], dtype=torch.bfloat16)
    bias = torch.tensor([0.25, -0.5, 1.0], dtype=torch.bfloat16)

    actual = direct_packed_int8_linear_reference(inputs.bfloat16(), packed, scale, bias, 3, 5)
    expected = _expected_direct_projection(inputs.bfloat16(), codes, scale, bias)

    assert torch.equal(actual, expected)


def test_reference_from_quantized_uses_only_tiled_ternary_weights() -> None:
    rows, columns = 257, 5
    codes = (torch.arange(rows * columns, dtype=torch.int8).remainder(3) - 1).view(rows, columns)
    packed = _pack_codes(codes)
    inputs = torch.linspace(-2, 2, 3 * columns, dtype=torch.bfloat16).view(3, columns)
    quantized, activation_scale = bitvla_quantize_activation_int8(inputs)
    inverse_scale = activation_scale.reciprocal()
    scale = torch.tensor(0.75, dtype=torch.bfloat16)

    actual = packed_ternary_int8_linear_reference(
        quantized,
        inverse_scale,
        packed,
        scale,
        None,
        rows,
        columns,
        output_dtype=inputs.dtype,
    )
    expected = _expected_direct_projection(inputs, codes, scale, None)

    assert torch.equal(actual, expected)


def test_direct_kernel_auto_falls_back_to_reference_on_cpu() -> None:
    codes = torch.tensor([[-1, 0, 1, -1], [1, 1, 0, -1]], dtype=torch.int8)
    packed = _pack_codes(codes)
    inputs = torch.tensor([[0.5, -1.0, 2.0, 0.125]], dtype=torch.bfloat16)
    scale = torch.tensor(0.5, dtype=torch.bfloat16)

    actual = direct_packed_int8_linear(inputs, packed, scale, None, 2, 4)
    expected = direct_packed_int8_linear_reference(inputs, packed, scale, None, 2, 4)

    assert torch.equal(actual, expected)


def test_bf16_candidate_reference_matches_upstream_bf16_linear_without_full_weight_decode() -> None:
    torch.manual_seed(19)
    rows, columns = 17, 19
    codes = torch.randint(-1, 2, (rows, columns), dtype=torch.int8)
    packed = _pack_codes(codes)
    inputs = torch.randn(3, 2, columns, dtype=torch.bfloat16)
    scale = torch.tensor(0.140625, dtype=torch.float32)
    bias = torch.randn(rows, dtype=torch.bfloat16)

    actual = direct_packed_bf16_linear(
        inputs,
        packed,
        scale,
        bias,
        rows,
        columns,
        backend="reference",
        activation_backend="torch",
    )
    expected = torch.nn.functional.linear(
        quantize_activation(inputs),
        dequantize_packed_weight(packed, scale, rows, columns).to(inputs.dtype),
        bias,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("packed_shape", "scale_shape", "bias_shape", "message"),
    [
        ((1,), (1,), None, "exactly four"),
        ((2,), (3,), None, "weight scale"),
        ((2,), (1,), (3,), "bias"),
    ],
)
def test_direct_projection_validates_shapes(
    packed_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    bias_shape: tuple[int, ...] | None,
    message: str,
) -> None:
    inputs = torch.ones(1, 4, dtype=torch.bfloat16)
    packed = torch.zeros(packed_shape, dtype=torch.uint8)
    scale = torch.ones(scale_shape, dtype=torch.bfloat16)
    bias = None if bias_shape is None else torch.ones(bias_shape, dtype=torch.bfloat16)

    with pytest.raises((TypeError, ValueError), match=message):
        direct_packed_int8_linear_reference(inputs, packed, scale, bias, 2, 4)


@pytest.mark.skipif(not triton_packed_kernel_available(), reason="Triton CUDA kernel is unavailable")
def test_triton_direct_packed_kernel_matches_reference_without_weight_materialization() -> None:
    torch.manual_seed(17)
    rows, columns = 65, 37
    codes = torch.randint(-1, 2, (rows, columns), dtype=torch.int8, device="cuda")
    packed = _pack_codes(codes)
    inputs = torch.randn(3, 2, columns, dtype=torch.bfloat16, device="cuda")
    scale = torch.rand(rows, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(rows, dtype=torch.bfloat16, device="cuda")

    expected = direct_packed_int8_linear(inputs, packed, scale, bias, rows, columns, backend="reference")
    actual = direct_packed_int8_linear(inputs, packed, scale, bias, rows, columns, backend="triton")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not triton_packed_kernel_available(), reason="Triton CUDA kernel is unavailable")
def test_triton_dp4a_kernel_matches_integer_reference_with_scale_and_bias() -> None:
    torch.manual_seed(23)
    rows, columns = 35, 64
    codes = torch.randint(-1, 2, (rows, columns), dtype=torch.int8, device="cuda")
    row_major = _pack_codes(codes)
    interleaved = repack_packed_ternary_for_dp4a(row_major, rows, columns)
    inputs = torch.randn(3, 2, columns, dtype=torch.bfloat16, device="cuda")
    scale = torch.rand(rows, dtype=torch.float32, device="cuda")
    bias = torch.randn(rows, dtype=torch.bfloat16, device="cuda")

    expected = direct_packed_int8_linear_reference(
        inputs,
        row_major,
        scale,
        bias,
        rows,
        columns,
    )
    actual = direct_packed_dp4a_int8_linear(
        inputs,
        interleaved,
        scale,
        bias,
        rows,
        columns,
        backend="triton",
        activation_backend="hybrid",
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not triton_packed_kernel_available(), reason="Triton CUDA kernel is unavailable")
def test_triton_tensorcore_kernel_consumes_dp4a_layout_without_repacking() -> None:
    torch.manual_seed(31)
    rows, columns = 67, 64
    codes = torch.randint(-1, 2, (rows, columns), dtype=torch.int8, device="cuda")
    row_major = _pack_codes(codes)
    interleaved = repack_packed_ternary_for_dp4a(row_major, rows, columns)
    inputs = torch.randn(17, columns, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(0.125, dtype=torch.float32, device="cuda")

    expected = direct_packed_int8_linear_reference(
        inputs,
        row_major,
        scale,
        None,
        rows,
        columns,
    )
    actual = direct_packed_w2a8_linear(
        inputs,
        interleaved,
        scale,
        None,
        rows,
        columns,
        backend="triton",
        activation_backend="hybrid",
        small_batch_threshold=8,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not triton_packed_kernel_available(), reason="Triton CUDA kernel is unavailable")
def test_triton_activation_prepass_is_finite_and_bf16_candidate_matches_with_exact_activation() -> None:
    torch.manual_seed(29)
    rows, columns = 65, 37
    codes = torch.randint(-1, 2, (rows, columns), dtype=torch.int8, device="cuda")
    packed = _pack_codes(codes)
    inputs = torch.randn(3, 2, columns, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(0.28125, dtype=torch.float32, device="cuda")
    bias = torch.randn(rows, dtype=torch.bfloat16, device="cuda")

    expected_codes, expected_scale = bitvla_quantize_activation_int8(inputs)
    actual_codes, actual_scale = triton_bitvla_quantize_activation_int8(inputs)
    assert actual_codes.dtype == torch.int8
    assert actual_codes.shape == expected_codes.shape
    assert actual_scale.dtype == torch.float32
    assert actual_scale.shape == expected_scale.shape
    assert torch.isfinite(actual_scale).all()

    hybrid_codes, hybrid_scale = hybrid_bitvla_quantize_activation_int8(inputs)
    assert torch.equal(hybrid_scale, expected_scale)
    assert torch.equal(hybrid_codes, expected_codes)

    actual = direct_packed_bf16_linear(
        inputs,
        packed,
        scale,
        bias,
        rows,
        columns,
        backend="triton",
        activation_backend="torch",
    )
    expected = torch.nn.functional.linear(
        quantize_activation(inputs),
        dequantize_packed_weight(packed, scale, rows, columns).to(inputs.dtype),
        bias,
    )
    assert torch.equal(actual, expected)
