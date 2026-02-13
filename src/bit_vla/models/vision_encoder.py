"""Vision encoder for the canonical 1-bit-ready VLA model."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear


@dataclass
class VisionEncoderConfig:
    input_channels: int = 3
    hidden_dim: int = 256
    output_dim: int = 512
    dropout: float = 0.1


class VisionEncoder(nn.Module):
    """CNN vision encoder with quantized projection heads."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = VisionEncoderConfig(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.feature_proj = BitLinear(128 * 4 * 4, hidden_dim)
        self.output_proj = BitLinear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 5:
            # [B, T, C, H, W] -> use the latest frame for policy prediction.
            images = images[:, -1]
        x = self.backbone(images).flatten(1)
        x = self.dropout(F.relu(self.feature_proj(x)))
        return self.output_proj(x)

    def get_feature_shapes(self, input_shape: Tuple[int, int, int]) -> Dict[str, Tuple[int, ...]]:
        c, h, w = input_shape
        h1, w1 = (h + 2 * 2 - 5) // 2 + 1, (w + 2 * 2 - 5) // 2 + 1
        h2, w2 = (h1 + 2 - 3) // 2 + 1, (w1 + 2 - 3) // 2 + 1
        h3, w3 = (h2 + 2 - 3) // 2 + 1, (w2 + 2 - 3) // 2 + 1
        return {
            "input": (c, h, w),
            "after_conv1": (32, h1, w1),
            "after_conv2": (64, h2, w2),
            "after_conv3": (128, h3, w3),
            "after_pool": (128, 4, 4),
            "flattened": (128 * 4 * 4,),
        }