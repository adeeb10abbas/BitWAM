import pytest
import torch
from torch import nn

from lerobot_policy_bitwam.bitvla_packed_kernel import (
    direct_packed_dp4a_int8_linear,
    direct_packed_int8_linear_reference,
    repack_packed_ternary_for_dp4a,
)
from lerobot_policy_bitwam.bitvla_w2a8_training import (
    enable_w2a8_qat_semantics,
    w2a8_ste_linear,
)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    values = codes.flatten().add(1).to(torch.uint8).view(-1, 4)
    return values[:, 0] | values[:, 1] << 2 | values[:, 2] << 4 | values[:, 3] << 6


def test_w2a8_ste_forward_matches_deployment_integer_reference() -> None:
    inputs = torch.tensor(
        [[0.5, -1.0, 2.0, 0.125], [-0.25, 0.75, 1.5, -2.0]],
        dtype=torch.bfloat16,
    )
    weight = torch.tensor(
        [[-0.8, 0.05, 0.9, -0.4], [0.2, 0.7, -0.6, -0.1], [1.0, -0.3, 0.1, 0.5]],
        dtype=torch.bfloat16,
    )
    bias = torch.tensor([0.25, -0.5, 0.125], dtype=torch.bfloat16)
    step = weight.float().abs().mean().clamp(min=1e-5)
    codes = (weight.float() / step).round().clamp(-1, 1).to(torch.int8)

    actual = w2a8_ste_linear(inputs, weight, bias)
    expected = direct_packed_int8_linear_reference(
        inputs,
        _pack_codes(codes),
        step,
        bias,
        weight.shape[0],
        weight.shape[1],
    )

    assert torch.equal(actual, expected)


def test_w2a8_ste_backward_matches_fake_quantized_linear_surrogate() -> None:
    torch.manual_seed(41)
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    weight = torch.randn(5, 8, requires_grad=True)
    bias = torch.randn(5, requires_grad=True)
    external_gradient = torch.randn(2, 3, 5)

    actual = w2a8_ste_linear(inputs, weight, bias)
    actual.backward(external_gradient)
    actual_gradients = (inputs.grad.clone(), weight.grad.clone(), bias.grad.clone())

    inputs.grad = weight.grad = bias.grad = None
    input_values = inputs.float()
    input_scale = 127 / input_values.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    dequantized_inputs = (
        (input_values * input_scale).round().clamp(-128, 127) / input_scale
    ).to(inputs.dtype)
    weight_values = weight.float()
    weight_step = weight_values.abs().mean().clamp(min=1e-5)
    dequantized_weight = (
        (weight_values / weight_step).round().clamp(-1, 1) * weight_step
    ).to(weight.dtype)
    input_proxy = inputs + (dequantized_inputs - inputs).detach()
    weight_proxy = weight + (dequantized_weight - weight).detach()
    surrogate = torch.nn.functional.linear(input_proxy, weight_proxy, bias)
    surrogate.backward(external_gradient)

    torch.testing.assert_close(actual_gradients[0], inputs.grad)
    torch.testing.assert_close(actual_gradients[1], weight.grad)
    torch.testing.assert_close(actual_gradients[2], bias.grad)


def test_enable_w2a8_qat_semantics_patches_only_one_bit_linears() -> None:
    bitlinear_type = type(
        "BitLinear",
        (nn.Linear,),
        {"__module__": "transformers.models.llava.modeling_bitnet"},
    )
    ternary = bitlinear_type(8, 4)
    ternary.weight_bits = 1
    ternary.input_bits = 8
    full_precision = bitlinear_type(8, 4)
    full_precision.weight_bits = 32
    full_precision.input_bits = 8
    model = nn.ModuleList([ternary, full_precision])

    converted = enable_w2a8_qat_semantics(model, scope="text")

    assert converted == 1
    assert ternary.w2a8_qat_semantics is True
    assert not hasattr(full_precision, "w2a8_qat_semantics")
    output = ternary(torch.randn(2, 8))
    assert output.shape == (2, 4)


def test_enable_w2a8_qat_semantics_rejects_unknown_activation_backend() -> None:
    with pytest.raises(ValueError, match="activation backend"):
        enable_w2a8_qat_semantics(nn.Linear(2, 2), activation_backend="unknown")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_w2a8_ste_uses_integer_forward_and_propagates_finite_gradients() -> None:
    torch.manual_seed(47)
    rows, columns = 48, 64
    inputs = torch.randn(32, columns, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        rows,
        columns,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    bias = torch.randn(rows, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    step = weight.float().abs().mean().clamp(min=1e-5)
    codes = (weight.float() / step).round().clamp(-1, 1).to(torch.int8)

    actual = w2a8_ste_linear(inputs, weight, bias)
    expected = direct_packed_int8_linear_reference(
        inputs,
        _pack_codes(codes),
        step,
        bias,
        rows,
        columns,
    )

    assert torch.equal(actual, expected)
    actual.float().square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert bias.grad is not None and torch.isfinite(bias.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_w2a8_ste_triton_activation_matches_packed_deployment() -> None:
    torch.manual_seed(53)
    rows, columns = 48, 64
    inputs = torch.randn(17, columns, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        rows,
        columns,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    bias = torch.randn(rows, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    step = weight.float().abs().mean().clamp(min=1e-5)
    codes = (weight.float() / step).round().clamp(-1, 1).to(torch.int8)
    packed = repack_packed_ternary_for_dp4a(_pack_codes(codes), rows, columns)

    actual = w2a8_ste_linear(inputs, weight, bias, activation_backend="triton")
    expected = direct_packed_dp4a_int8_linear(
        inputs,
        packed,
        step,
        bias,
        rows,
        columns,
        backend="triton",
        activation_backend="triton",
    )

    assert torch.equal(actual, expected)
    actual.float().square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert bias.grad is not None and torch.isfinite(bias.grad).all()
