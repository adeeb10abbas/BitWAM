"""
VLABitNet: Complete Vision-Language-Action model with 1.58-bit quantization.

This is the main model that combines vision encoding, language encoding, 
and action decoding using BitNet quantization principles.
"""

import torch
import torch.nn as nn
from typing import Optional

from .vision_encoder import VisionEncoder
from .language_encoder import LanguageEncoder
from .action_decoder import ActionDecoder
from .bitlinear import BitLinear


class VLABitNet(nn.Module):
    """
    Complete VLA model with BitNet quantization.
    
    Combines vision encoder, language encoder, and action decoder with 
    multimodal fusion using BitLinear layers.
    
    Args:
        vision_input_channels: Number of vision input channels
        vocab_size: Size of language vocabulary
        hidden_dim: Hidden dimension for feature processing
        action_dim: Output action dimensionality
        
    Example:
        >>> model = VLABitNet(hidden_dim=512, action_dim=7)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> tokens = torch.randint(0, 1000, (4, 32))
        >>> actions = model(images, tokens)
        >>> print(actions.shape)  # torch.Size([4, 7])
    """
    
    def __init__(
        self, 
        vision_input_channels: int = 3,
        vocab_size: int = 10000,
        hidden_dim: int = 512,
        action_dim: int = 7,
        use_tanh_actions: bool = True
    ):
        super().__init__()
        
        # Individual encoders
        self.vision_encoder = VisionEncoder(
            input_channels=vision_input_channels,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        self.language_encoder = LanguageEncoder(
            vocab_size=vocab_size,
            embed_dim=hidden_dim // 2,
            hidden_dim=hidden_dim,
            max_seq_len=128
        )
        
        # Multimodal fusion with BitLinear
        self.fusion_layer = BitLinear(hidden_dim * 2, hidden_dim)
        
        # Action decoder
        self.action_decoder = ActionDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            use_tanh=use_tanh_actions
        )
        
    def forward(
        self, 
        images: torch.Tensor, 
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the complete VLA model.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            token_ids: Language token IDs [batch_size, seq_len]
            attention_mask: Language attention mask [batch_size, seq_len]
            
        Returns:
            Action predictions [batch_size, action_dim]
        """
        # Encode vision and language separately
        visual_features = self.vision_encoder(images)
        language_features = self.language_encoder(token_ids, attention_mask)
        
        # Fuse multimodal features
        combined_features = torch.cat(
            [visual_features, language_features], dim=1
        )
        fused_features = self.fusion_layer(combined_features)
        
        # Decode to actions
        actions = self.action_decoder(fused_features)
        
        return actions
    
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode only vision inputs."""
        return self.vision_encoder(images)
    
    def encode_language(
        self, 
        token_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode only language inputs."""
        return self.language_encoder(token_ids, attention_mask)
    
    def get_quantization_summary(self) -> dict:
        """
        Get summary of quantization across the entire model.
        
        Returns:
            Dictionary with comprehensive quantization statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        
        # Count BitLinear parameters
        bitlinear_params = 0
        bitlinear_layers = 0
        
        for name, module in self.named_modules():
            if hasattr(module, 'quantize_weights'):  # BitLinear layer
                bitlinear_layers += 1
                bitlinear_params += sum(
                    p.numel() for p in module.parameters()
                )
        
        fp16_params = total_params - bitlinear_params
        
        return {
            "total_parameters": total_params,
            "bitlinear_parameters": bitlinear_params,
            "fp16_parameters": fp16_params,
            "bitlinear_layers": bitlinear_layers,
            "quantized_ratio": bitlinear_params / total_params * 100,
            "estimated_size_mb": {
                "fp32": total_params * 4 / 1024**2,
                "quantized": (
                    bitlinear_params * 0.2 + fp16_params * 4
                ) / 1024**2,
            },
            "memory_savings": {
                # MB saved
                "absolute": (bitlinear_params * 3.8) / 1024**2,  
                # % saved
                "percentage": (bitlinear_params / total_params) * 95,  
            }
        } 