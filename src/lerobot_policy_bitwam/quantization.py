"""Training-time ternarization for eligible BitWAM linear layers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

QuantizationScope = Literal["none", "qwen", "qwen_dit"]
QATRecovery = Literal["none", "qwen_edges", "dit_tail4"]


class TernaryLinear(nn.Module):
    """Linear layer with BF16 masters and straight-through simulated quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=torch.bfloat16)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device, dtype=torch.bfloat16)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = self.in_features**-0.5
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> TernaryLinear:
        """Copy a linear layer into BF16 master parameters without changing shape."""
        converted = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
        )
        with torch.no_grad():
            converted.weight.copy_(linear.weight.detach().to(torch.bfloat16))
            if converted.bias is not None and linear.bias is not None:
                converted.bias.copy_(linear.bias.detach().to(torch.bfloat16))
        converted.train(linear.training)
        return converted

    def quantized_weight(self) -> tuple[Tensor, Tensor]:
        """Return ternary codes and per-output-channel BF16 abs-mean scales."""
        scale = (
            self.weight.detach().abs().mean(dim=1, keepdim=True).clamp_min(torch.finfo(torch.bfloat16).tiny)
        )
        ternary = torch.round(self.weight.detach() / scale).clamp_(-1, 1)
        return ternary, scale

    @staticmethod
    def simulate_int8_activations(activations: Tensor) -> tuple[Tensor, Tensor]:
        """Simulate symmetric per-token INT8 activations with a straight-through path."""
        scale = activations.detach().float().abs().amax(dim=-1, keepdim=True).div(127).clamp_min(1e-8)
        quantized = torch.round(activations.float() / scale).clamp_(-127, 127)
        dequantized = (quantized * scale).to(activations.dtype)
        return activations + (dequantized - activations).detach(), scale

    def forward(self, activations: Tensor) -> Tensor:
        quantized_activations, _ = self.simulate_int8_activations(activations)
        ternary, scale = self.quantized_weight()
        dequantized_weight = ternary * scale
        ste_weight = self.weight + (dequantized_weight - self.weight).detach()
        output = F.linear(
            quantized_activations.to(self.weight.dtype),
            ste_weight,
            self.bias,
        )
        return output.to(activations.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        )


@dataclass(frozen=True)
class QuantizationReport:
    """Parameter and module coverage from deterministic QAT conversion."""

    scope: QuantizationScope
    total_parameter_count: int
    eligible_parameter_count: int
    ternary_parameter_count: int
    bf16_parameter_count: int
    converted_modules: tuple[str, ...]
    recovery: QATRecovery = "none"
    recovery_bf16_modules: tuple[str, ...] = ()

    @property
    def eligible_ternary_fraction(self) -> float:
        if self.eligible_parameter_count == 0:
            return 0.0
        return self.ternary_parameter_count / self.eligible_parameter_count

    def to_dict(self) -> dict:
        return asdict(self) | {"eligible_ternary_fraction": self.eligible_ternary_fraction}


def _eligible_kind(module_name: str) -> str | None:
    qwen_layer = ".qwen.model.model.language_model.layers." in f".{module_name}."
    if qwen_layer and (".self_attn." in f".{module_name}." or ".mlp." in f".{module_name}."):
        return "qwen"

    dit_block = ".action_model.model.transformer_blocks." in f".{module_name}."
    if dit_block and (".attn1." in f".{module_name}." or ".ff." in f".{module_name}."):
        return "dit"
    return None


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, child_name = name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def _recovery_exclusions(candidates: list[tuple[str, nn.Linear]], recovery: QATRecovery) -> set[str]:
    if recovery == "none":
        return set()
    pattern = (
        r"\.language_model\.layers\.(\d+)\." if recovery == "qwen_edges" else r"\.transformer_blocks\.(\d+)\."
    )
    indexed = [(name, int(match.group(1))) for name, _ in candidates if (match := re.search(pattern, name))]
    if not indexed:
        return set()
    indices = sorted({index for _, index in indexed})
    excluded_indices = {indices[0], indices[-1]} if recovery == "qwen_edges" else set(indices[-4:])
    return {name for name, index in indexed if index in excluded_indices}


def convert_for_qat(
    policy: nn.Module,
    scope: QuantizationScope,
    *,
    recovery: QATRecovery = "none",
) -> QuantizationReport:
    """Deterministically replace eligible linears for the requested QAT scope."""
    if scope not in {"none", "qwen", "qwen_dit"}:
        raise ValueError(f"Unknown quantization scope: {scope}")
    if recovery not in {"none", "qwen_edges", "dit_tail4"}:
        raise ValueError(f"Unknown QAT recovery: {recovery}")
    if recovery == "qwen_edges" and scope != "qwen":
        raise ValueError("qwen_edges recovery requires qwen scope.")
    if recovery == "dit_tail4" and scope != "qwen_dit":
        raise ValueError("dit_tail4 recovery requires qwen_dit scope.")

    total = sum(parameter.numel() for parameter in policy.parameters())
    selected_kinds = {"qwen"} if scope == "qwen" else {"qwen", "dit"} if scope == "qwen_dit" else set()
    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in sorted(policy.named_modules(), key=lambda item: item[0]):
        if isinstance(module, nn.Linear) and _eligible_kind(name) in selected_kinds:
            candidates.append((name, module))

    excluded = _recovery_exclusions(candidates, recovery)
    selected = [(name, module) for name, module in candidates if name not in excluded]
    eligible = sum(module.weight.numel() for _, module in candidates)
    for name, module in selected:
        _replace_module(policy, name, TernaryLinear.from_linear(module))

    ternary = sum(module.weight.numel() for module in policy.modules() if isinstance(module, TernaryLinear))
    report = QuantizationReport(
        scope=scope,
        total_parameter_count=total,
        eligible_parameter_count=eligible,
        ternary_parameter_count=ternary,
        bf16_parameter_count=total - ternary,
        converted_modules=tuple(name for name, _ in selected),
        recovery=recovery,
        recovery_bf16_modules=tuple(sorted(excluded)),
    )
    if hasattr(policy, "config"):
        policy.config.quantization_scope = scope
    return report
