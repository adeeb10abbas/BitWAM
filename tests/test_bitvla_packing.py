import torch
from torch import nn

from lerobot_policy_bitwam.bitvla_packing import (
    dequantize_packed_weight,
    pack_bitlinear_weights,
    quantize_activation,
    quantize_activation_int8,
    unpack_packed_weight_int8_transposed,
)


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, weight_bits: int = 1) -> None:
        super().__init__(in_features, out_features, bias=False, dtype=torch.bfloat16)
        self.weight_bits = weight_bits

    def quantize_weights(self) -> None:
        values = self.weight.detach().flatten()
        codes = values.sign().to(torch.int8).add(1).to(torch.uint8)
        padding = (-codes.numel()) % 4
        codes = torch.nn.functional.pad(codes, (0, padding)).view(-1, 4)
        packed = codes[:, 0] | codes[:, 1] << 2 | codes[:, 2] << 4 | codes[:, 3] << 6
        self.register_buffer("q_weight", packed)
        self.register_parameter("weight", None)


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ternary = BitLinear(5, 3)
        self.full_precision = BitLinear(5, 3, weight_bits=32)
        self.output = nn.Linear(3, 2, bias=False)


def test_pack_bitlinear_weights_replaces_only_one_bit_matrices() -> None:
    policy = FakePolicy()
    report = pack_bitlinear_weights(policy)

    assert report.packed_layers == 1
    assert report.scope == "all"
    assert report.packed_weight_count == 15
    assert report.bf16_weight_bytes_replaced == 30
    assert report.packed_weight_bytes == 4
    assert report.weight_storage_reduction == 1 - 4 / 30
    assert policy.ternary.weight is None
    assert policy.ternary.q_weight.dtype == torch.uint8
    assert policy.full_precision.weight is not None


def test_pack_bitlinear_weights_validates_scope() -> None:
    policy = FakePolicy()
    try:
        pack_bitlinear_weights(policy, scope="unknown")
    except ValueError as error:
        assert "Unsupported" in str(error)
    else:
        raise AssertionError("packing should reject an unknown scope")


def test_pack_bitlinear_weights_rejects_models_without_eligible_layers() -> None:
    policy = nn.Sequential(nn.Linear(2, 2))
    try:
        pack_bitlinear_weights(policy)
    except RuntimeError as error:
        assert "No eligible" in str(error)
    else:
        raise AssertionError("packing should fail when a model has no BitLinear layers")


def test_dequantize_packed_weight_restores_four_codes_per_byte() -> None:
    codes = torch.tensor([-1, 0, 1, -1, 1, 0], dtype=torch.int8)
    shifted = torch.nn.functional.pad(codes.add(1).to(torch.uint8), (0, 2)).view(-1, 4)
    packed = shifted[:, 0] | shifted[:, 1] << 2 | shifted[:, 2] << 4 | shifted[:, 3] << 6

    restored = dequantize_packed_weight(packed, torch.tensor(0.5), 2, 3)

    assert restored.dtype == torch.bfloat16
    assert torch.equal(restored, codes.view(2, 3).to(torch.bfloat16) * 0.5)


def test_unpack_int8_transposes_for_native_matrix_multiply() -> None:
    codes = torch.tensor([-1, 0, 1, -1, 1, 0], dtype=torch.int8)
    shifted = torch.nn.functional.pad(codes.add(1).to(torch.uint8), (0, 2)).view(-1, 4)
    packed = shifted[:, 0] | shifted[:, 1] << 2 | shifted[:, 2] << 4 | shifted[:, 3] << 6

    restored = unpack_packed_weight_int8_transposed(packed, 2, 3)

    assert restored.dtype == torch.int8
    assert restored.is_contiguous()
    assert torch.equal(restored, codes.view(2, 3).T)


def test_activation_quantization_returns_int8_and_inverse_scale() -> None:
    inputs = torch.tensor([[0.0, -1.0, 0.5], [2.0, -0.5, 1.0]], dtype=torch.bfloat16)
    quantized, inverse_scale = quantize_activation_int8(inputs)

    assert quantized.dtype == torch.int8
    assert inverse_scale.dtype == torch.float32
    reconstructed = quantized.float() * inverse_scale
    assert torch.allclose(reconstructed, inputs.float(), atol=0.01)


def test_activation_quantization_matches_bitvla_formula() -> None:
    inputs = torch.tensor([[0.0, -1.0, 0.5], [2.0, -0.5, 1.0]], dtype=torch.bfloat16)
    values = inputs.float()
    scale = 127 / values.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    expected = ((values * scale).round().clamp(-128, 127) / scale).to(inputs.dtype)

    assert torch.equal(quantize_activation(inputs), expected)
