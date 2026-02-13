"""
BitLinear: Core quantized linear layer for 1-bit neural networks.

This module implements the fundamental BitLinear layer that replaces 
standard nn.Linear layers in BitNet architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Module):
    """
    BitNet Linear Layer: Core building block with ternary weights and 8-bit 
    activations.
    
    This replaces standard nn.Linear with quantized computation:
    1. Weights are quantized to {-1, 0, +1} during forward pass
    2. Activations are quantized to 8-bit integers  
    3. Computation uses quantized values but maintains gradients for training
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term (default: False for BitNet)
        eps: Small epsilon for numerical stability
        
    Example:
        >>> layer = BitLinear(256, 512)
        >>> x = torch.randn(4, 256)
        >>> out = layer(x)
        >>> print(out.shape)  # torch.Size([4, 512])
        >>> stats = layer.get_quantization_stats()
        >>> print(stats['unique_weight_values'])  # Should be [-1, 0, 1]
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = False,
        eps: float = 1e-5
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        # Initialize weights (will be quantized during forward pass)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )
        
        # BitNet typically doesn't use bias for simplicity
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Store quantized versions for inspection
        self.quantized_weight = None
        self.weight_scale = None
        self.activation_scale = None
        
    def quantize_weights(
        self, 
        weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights using absmean method.
        
        Args:
            weights: Input weight tensor
            
        Returns:
            Tuple of (quantized_weights, scale_factor)
        """
        scale = torch.mean(torch.abs(weights)) + self.eps
        quantized = torch.sign(weights) * (torch.abs(weights) > scale).float()
        return quantized, scale
        
    def quantize_activations(
        self, 
        activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Simulate 8-bit activation quantization.
        
        During training, we simulate quantization but keep gradients flowing.
        During inference, this could be replaced with actual int8 computation.
        
        Args:
            activations: Input activation tensor
            
        Returns:
            Quantized activations (still as float for gradient flow)
        """
        if not self.training:
            return activations
            
        # Get scale factor
        scale = torch.max(torch.abs(activations)) + self.eps
        
        # Simulate 8-bit quantization effect but keep as float
        x_quantized = torch.round(activations / scale * 127) / 127 * scale
        
        # Store for analysis
        self.activation_scale = scale.detach()
        
        return x_quantized
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with BitNet quantization.
        
        Args:
            x: Input tensor [batch_size, ..., in_features]
            
        Returns:
            Output tensor [batch_size, ..., out_features]
        """
        # 1. Quantize weights to {-1, 0, +1}
        quantized_weight, weight_scale = self.quantize_weights(self.weight)
        
        # Store for inspection
        self.quantized_weight = quantized_weight.detach()
        self.weight_scale = weight_scale.detach()
        
        # 2. Quantize activations to simulate 8-bit
        x_quantized = self.quantize_activations(x)
            
        # 3. Matrix multiplication with quantized values
        output = F.linear(x_quantized, quantized_weight, self.bias)
        
        return output
    
    def get_quantization_stats(self) -> dict:
        """
        Get statistics about quantization for analysis.
        
        Returns:
            Dictionary with quantization statistics
        """
        if self.quantized_weight is None:
            return {"message": "Run forward pass first"}
            
        unique_values = torch.unique(self.quantized_weight)
        sparsity = (self.quantized_weight == 0).float().mean()
        
        stats = {
            "unique_weight_values": unique_values.tolist(),
            "weight_sparsity": sparsity.item(),
            "weight_scale": self.weight_scale.item(),
            "expected_values": "Should be close to [-1, 0, 1]",
            "parameter_count": self.weight.numel(),
            "effective_bits": 1.58,  # BitNet 1.58-bit
        }
        
        if self.activation_scale is not None:
            stats["activation_scale"] = self.activation_scale.item()
            
        return stats

    def get_quantization_summary(self) -> dict:
        """
        Return a model-level style summary for compatibility with analysis helpers.
        """
        total_parameters = sum(p.numel() for p in self.parameters())
        # BitLinear stores all linear weights in ternary-ready form.
        quantized_parameters = self.weight.numel()
        fp_parameters = total_parameters - quantized_parameters
        quantized_ratio = (
            (quantized_parameters / total_parameters) * 100
            if total_parameters > 0
            else 0.0
        )
        return {
            "total_parameters": total_parameters,
            "quantized_parameters": quantized_parameters,
            "bitlinear_parameters": quantized_parameters,
            "fp_parameters": fp_parameters,
            "quantized_ratio": quantized_ratio,
        }
        
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'quantization=1.58bit'
        ) 