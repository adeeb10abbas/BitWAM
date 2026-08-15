import torch
from torch import nn

from lerobot_policy_bitwam.bitvla_packing import dequantize_packed_weight, pack_bitlinear_weights


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
    assert report.packed_weight_count == 15
    assert report.bf16_weight_bytes_replaced == 30
    assert report.packed_weight_bytes == 4
    assert report.weight_storage_reduction == 1 - 4 / 30
    assert policy.ternary.weight is None
    assert policy.ternary.q_weight.dtype == torch.uint8
    assert policy.full_precision.weight is not None


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
