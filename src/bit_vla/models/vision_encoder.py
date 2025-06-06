"""
Vision encoder for 1-bit VLA models.

This module implements a vision encoder that processes RGB images into 
feature representations using BitNet quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear


class VisionEncoder(nn.Module):
    """
    Simple vision encoder using BitNet principles.
    
    Processes RGB images to feature representations using a simple CNN 
    backbone followed by BitLinear layers for feature processing.
    
    Args:
        input_channels: Number of input channels (default: 3 for RGB)
        hidden_dim: Hidden dimension for feature processing
        output_dim: Output feature dimension
        
    Example:
        >>> encoder = VisionEncoder(output_dim=512)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> features = encoder(images)
        >>> print(features.shape)  # torch.Size([4, 512])
    """
    
    def __init__(
        self, 
        input_channels: int = 3, 
        hidden_dim: int = 256, 
        output_dim: int = 512
    ):
        super().__init__()
        
        # Simple CNN backbone - could be replaced with quantized ResNet/ViT
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3
        )
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # BitNet quantized layers for feature processing
        self.feature_proj = BitLinear(128 * 8 * 8, hidden_dim)
        self.output_proj = BitLinear(hidden_dim, output_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision encoder.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            
        Returns:
            Visual features [batch_size, output_dim]
        """
        # CNN feature extraction
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.flatten(1)  # [batch_size, 128*8*8]
        
        # BitNet quantized processing
        x = F.relu(self.feature_proj(x))
        visual_features = self.output_proj(x)
        
        return visual_features
    
    def get_feature_shapes(self, input_shape: tuple) -> dict:
        """
        Get intermediate feature shapes for debugging.
        
        Args:
            input_shape: Input image shape (C, H, W)
            
        Returns:
            Dictionary with intermediate shapes
        """
        c, h, w = input_shape
        
        # After conv1: stride 2, kernel 7, padding 3
        h1, w1 = (h + 2*3 - 7) // 2 + 1, (w + 2*3 - 7) // 2 + 1
        
        # After conv2: stride 2, kernel 3, padding 1  
        h2, w2 = (h1 + 2*1 - 3) // 2 + 1, (w1 + 2*1 - 3) // 2 + 1
        
        return {
            "input": (c, h, w),
            "after_conv1": (64, h1, w1),
            "after_conv2": (128, h2, w2),
            "after_pool": (128, 8, 8),
            "flattened": 128 * 8 * 8,
        } 