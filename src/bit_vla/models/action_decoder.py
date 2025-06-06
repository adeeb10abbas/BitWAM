"""
Action decoder for 1-bit VLA models.

This module implements an action decoder that takes fused multimodal 
features and outputs continuous control actions using BitNet quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear


class ActionDecoder(nn.Module):
    """
    Action decoder using BitNet quantization.
    
    Takes fused multimodal features and decodes them into continuous 
    control actions through BitLinear layers.
    
    Args:
        input_dim: Input feature dimension (from fused features)
        hidden_dim: Hidden dimension for processing
        action_dim: Output action dimensionality
        use_tanh: Whether to apply tanh activation to actions
        
    Example:
        >>> decoder = ActionDecoder(input_dim=1024, action_dim=7)
        >>> features = torch.randn(4, 1024)
        >>> actions = decoder(features)
        >>> print(actions.shape)  # torch.Size([4, 7])
    """
    
    def __init__(
        self, 
        input_dim: int = 1024, 
        hidden_dim: int = 512, 
        action_dim: int = 7,
        use_tanh: bool = True
    ):
        super().__init__()
        self.use_tanh = use_tanh
        
        # BitNet layers for action decoding
        self.hidden_layer1 = BitLinear(input_dim, hidden_dim)
        self.hidden_layer2 = BitLinear(hidden_dim, hidden_dim // 2)
        
        # Final action prediction layer - keep as FP16 for precision
        self.action_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # Optional action bounds prediction
        self.bounds_head = nn.Linear(hidden_dim // 2, action_dim * 2)
        
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through action decoder.
        
        Args:
            fused_features: Fused multimodal features [batch_size, input_dim]
            
        Returns:
            Action predictions [batch_size, action_dim]
        """
        # BitNet hidden layers
        x = F.relu(self.hidden_layer1(fused_features))
        x = F.relu(self.hidden_layer2(x))
        
        # Final action prediction
        actions = self.action_head(x)
        
        # Apply tanh for bounded actions if requested
        if self.use_tanh:
            actions = torch.tanh(actions)
            
        return actions
    
    def forward_with_bounds(
        self, 
        fused_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with action bounds prediction.
        
        Args:
            fused_features: Fused multimodal features [batch_size, input_dim]
            
        Returns:
            Tuple of (actions, bounds) where bounds are 
            [batch_size, action_dim*2]
        """
        # Shared hidden processing
        x = F.relu(self.hidden_layer1(fused_features))
        x = F.relu(self.hidden_layer2(x))
        
        # Action and bounds prediction
        actions = self.action_head(x)
        bounds = self.bounds_head(x)
        
        # Apply tanh if requested
        if self.use_tanh:
            actions = torch.tanh(actions)
            
        return actions, bounds
    
    def get_action_statistics(self) -> dict:
        """
        Get statistics about the action decoder for analysis.
        
        Returns:
            Dictionary with decoder statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        bitlinear_params = sum(
            p.numel() for name, p in self.named_parameters() 
            if any(layer in name for layer in 
                   ['hidden_layer1', 'hidden_layer2'])
        )
        fp16_params = total_params - bitlinear_params
        
        return {
            "total_params": total_params,
            "bitlinear_params": bitlinear_params,
            "fp16_params": fp16_params,
            "quantized_ratio": bitlinear_params / total_params * 100,
            "use_tanh": self.use_tanh,
        } 