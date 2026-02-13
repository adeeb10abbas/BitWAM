"""Canonical Vision-Language-Action model with 1-bit-ready components."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_decoder import ActionDecoder
from .bitlinear import BitLinear
from .language_encoder import LanguageEncoder
from .vision_encoder import VisionEncoder


@dataclass
class VLABitNetConfig:
    vision_input_channels: int = 3
    vocab_size: int = 8192
    hidden_dim: int = 512
    action_dim: int = 7
    state_dim: int = 0
    max_seq_len: int = 64
    use_tanh_actions: bool = True
    dropout: float = 0.1


class VLABitNet(nn.Module):
    """
    Canonical multimodal model contract:
    - images: [B, C, H, W] or [B, T, C, H, W]
    - token_ids: [B, L]
    - attention_mask: [B, L] bool (optional)
    - states: [B, D] or [B, T, D] (optional)
    Returns:
    - actions: [B, action_dim]
    """

    def __init__(
        self,
        vision_input_channels: int = 3,
        vocab_size: int = 8192,
        hidden_dim: int = 512,
        action_dim: int = 7,
        state_dim: int = 0,
        max_seq_len: int = 64,
        use_tanh_actions: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = VLABitNetConfig(
            vision_input_channels=vision_input_channels,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            max_seq_len=max_seq_len,
            use_tanh_actions=use_tanh_actions,
            dropout=dropout,
        )

        self.vision_encoder = VisionEncoder(
            input_channels=vision_input_channels,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.language_encoder = LanguageEncoder(
            vocab_size=vocab_size,
            embed_dim=hidden_dim // 2,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.state_projector = (
            BitLinear(state_dim, hidden_dim) if state_dim > 0 else None
        )
        fusion_input_dim = hidden_dim * (3 if state_dim > 0 else 2)
        self.fusion_layer = BitLinear(fusion_input_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_dropout = nn.Dropout(dropout)
        self.action_decoder = ActionDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            use_tanh=use_tanh_actions,
            dropout=dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vision_features = self.vision_encoder(images)
        language_features = self.language_encoder(token_ids, attention_mask)

        features = [vision_features, language_features]
        if self.state_projector is not None:
            if states is None:
                raise ValueError("states must be provided when state_dim > 0")
            if states.dim() == 3:
                states = states[:, -1]
            features.append(self.state_projector(states))

        fused = torch.cat(features, dim=-1)
        fused = self.fusion_dropout(F.gelu(self.fusion_layer(fused)))
        fused = self.fusion_norm(fused)
        return self.action_decoder(fused)

    def forward_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self(
            images=batch["images"],
            token_ids=batch["token_ids"],
            attention_mask=batch.get("attention_mask"),
            states=batch.get("states"),
        )

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(images)

    def encode_language(
        self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.language_encoder(token_ids, attention_mask)

    def get_quantization_summary(self) -> dict:
        total_parameters = sum(p.numel() for p in self.parameters())
        bitlinear_parameters = 0
        bitlinear_layers = 0
        for module in self.modules():
            if isinstance(module, BitLinear):
                bitlinear_layers += 1
                bitlinear_parameters += sum(p.numel() for p in module.parameters())
        fp_parameters = total_parameters - bitlinear_parameters
        quantized_ratio = (
            (bitlinear_parameters / total_parameters) * 100
            if total_parameters > 0
            else 0.0
        )
        fp32_size_mb = total_parameters * 4 / 1024**2
        quantized_size_mb = (bitlinear_parameters * 0.2 + fp_parameters * 4) / 1024**2
        return {
            "total_parameters": total_parameters,
            "quantized_parameters": bitlinear_parameters,
            "bitlinear_parameters": bitlinear_parameters,
            "fp_parameters": fp_parameters,
            "bitlinear_layers": bitlinear_layers,
            "quantized_ratio": quantized_ratio,
            "estimated_size_mb": {
                "fp32": fp32_size_mb,
                "quantized": quantized_size_mb,
            },
            "memory_savings": {
                "absolute_mb": fp32_size_mb - quantized_size_mb,
                "percentage": ((fp32_size_mb - quantized_size_mb) / fp32_size_mb) * 100
                if fp32_size_mb > 0
                else 0.0,
            },
        }