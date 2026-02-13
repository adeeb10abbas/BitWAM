"""Action decoder for the canonical VLA model."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear


@dataclass
class ActionDecoderConfig:
    input_dim: int = 1024
    hidden_dim: int = 512
    action_dim: int = 7
    use_tanh: bool = True
    dropout: float = 0.1


class ActionDecoder(nn.Module):
    """Decodes fused features into a single control action."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        action_dim: int = 7,
        use_tanh: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = ActionDecoderConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            use_tanh=use_tanh,
            dropout=dropout,
        )
        self.use_tanh = use_tanh
        self.hidden_layer1 = BitLinear(input_dim, hidden_dim)
        self.hidden_layer2 = BitLinear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        # Keep the prediction head in full precision for training stability.
        self.action_head = nn.Linear(hidden_dim // 2, action_dim)
        self.bounds_head = nn.Linear(hidden_dim // 2, action_dim * 2)

    def _shared(self, fused_features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.gelu(self.hidden_layer1(fused_features)))
        return self.dropout(F.gelu(self.hidden_layer2(x)))

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        actions = self.action_head(self._shared(fused_features))
        if self.use_tanh:
            actions = torch.tanh(actions)
        return actions

    def forward_with_bounds(self, fused_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self._shared(fused_features)
        actions = self.action_head(shared)
        bounds = self.bounds_head(shared)
        if self.use_tanh:
            actions = torch.tanh(actions)
        return actions, bounds

    def get_action_statistics(self) -> Dict[str, float]:
        total_params = sum(p.numel() for p in self.parameters())
        bitlinear_params = sum(
            p.numel()
            for name, p in self.named_parameters()
            if any(name.startswith(prefix) for prefix in ("hidden_layer1", "hidden_layer2"))
        )
        fp_params = total_params - bitlinear_params
        quantized_ratio = (bitlinear_params / total_params) * 100 if total_params > 0 else 0.0
        return {
            "total_params": float(total_params),
            "bitlinear_params": float(bitlinear_params),
            "fp_params": float(fp_params),
            "quantized_ratio": quantized_ratio,
            "use_tanh": self.use_tanh,
        }