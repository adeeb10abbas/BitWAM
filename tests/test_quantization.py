"""Focused unit tests for BitWAM training-time ternarization."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from lerobot_policy_bitwam.quantization import TernaryLinear, convert_for_qat


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.out_proj = nn.Linear(4, 4)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8)
        self.down_proj = nn.Linear(8, 4)


class _QwenBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()
        self.norm_projection = nn.Linear(4, 4)


class _DiTBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = _Attention()
        self.ff = _MLP()
        self.norm_projection = nn.Linear(4, 4)


class _Tree(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen = nn.Module()
        self.qwen.model = nn.Module()
        self.qwen.model.model = nn.Module()
        self.qwen.model.model.language_model = nn.Module()
        self.qwen.model.model.language_model.layers = nn.ModuleList([_QwenBlock()])
        self.qwen.vision_projection = nn.Linear(4, 4)
        self.action_model = nn.Module()
        self.action_model.model = nn.Module()
        self.action_model.model.transformer_blocks = nn.ModuleList([_DiTBlock()])
        self.action_model.action_encoder = nn.Linear(4, 4)
        self.action_model.action_decoder = nn.Linear(4, 4)


def test_ternary_values_and_per_output_scales() -> None:
    layer = TernaryLinear(4, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, -1.0, 0.1, -0.1], [2.0, 0.0, -2.0, 0.0]]))
    ternary, scale = layer.quantized_weight()
    assert scale.shape == (2, 1)
    assert scale.dtype == torch.bfloat16
    assert set(ternary.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_activation_simulation_is_per_token_and_has_ste_gradient() -> None:
    activations = torch.randn(2, 3, 4, requires_grad=True)
    simulated, scale = TernaryLinear.simulate_int8_activations(activations)
    assert scale.shape == (2, 3, 1)
    simulated.sum().backward()
    torch.testing.assert_close(activations.grad, torch.ones_like(activations))


def test_ste_training_update_changes_bf16_master_weight() -> None:
    layer = TernaryLinear.from_linear(nn.Linear(4, 3))
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    before = layer.weight.detach().clone()
    output = layer(torch.randn(8, 4, dtype=torch.bfloat16))
    loss = output.float().square().mean()
    loss.backward()
    assert layer.weight.grad is not None
    assert torch.count_nonzero(layer.weight.grad) > 0
    optimizer.step()
    assert not torch.equal(layer.weight, before)


def test_qwen_scope_respects_conversion_boundaries() -> None:
    tree = _Tree()
    report = convert_for_qat(tree, "qwen")
    qwen_block = tree.qwen.model.model.language_model.layers[0]
    dit_block = tree.action_model.model.transformer_blocks[0]
    assert isinstance(qwen_block.self_attn.q_proj, TernaryLinear)
    assert isinstance(qwen_block.mlp.gate_proj, TernaryLinear)
    assert isinstance(qwen_block.norm_projection, nn.Linear)
    assert isinstance(tree.qwen.vision_projection, nn.Linear)
    assert isinstance(dit_block.attn1.q_proj, nn.Linear)
    assert report.ternary_parameter_count == sum(
        module.weight.numel() for module in tree.modules() if isinstance(module, TernaryLinear)
    )
    assert report.ternary_parameter_count == report.eligible_parameter_count


def test_qwen_dit_keeps_encoders_and_final_output_in_bf16() -> None:
    tree = _Tree()
    report = convert_for_qat(tree, "qwen_dit")
    dit_block = tree.action_model.model.transformer_blocks[0]
    assert isinstance(dit_block.attn1.q_proj, TernaryLinear)
    assert isinstance(dit_block.ff.gate_proj, TernaryLinear)
    assert isinstance(dit_block.norm_projection, nn.Linear)
    assert isinstance(tree.action_model.action_encoder, nn.Linear)
    assert isinstance(tree.action_model.action_decoder, nn.Linear)
    assert report.bf16_parameter_count + report.ternary_parameter_count == report.total_parameter_count


def test_ternary_layer_state_dict_round_trip() -> None:
    source = TernaryLinear.from_linear(nn.Linear(4, 3))
    restored = TernaryLinear(4, 3)
    restored.load_state_dict(deepcopy(source.state_dict()))
    assert torch.equal(restored.weight, source.weight)
    assert torch.equal(restored.bias, source.bias)


def test_none_scope_is_noop() -> None:
    tree = _Tree()
    before = tuple(type(module) for module in tree.modules())
    report = convert_for_qat(tree, "none")
    assert tuple(type(module) for module in tree.modules()) == before
    assert report.eligible_parameter_count == 0
    assert report.ternary_parameter_count == 0
