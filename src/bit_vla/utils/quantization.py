"""
BitNet quantization utilities for 1.58-bit weights and 8-bit activations.

This module contains the core quantization functions used throughout the 
1-bit VLA research codebase.
"""

import torch
from typing import Tuple


def absmean_quantize_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    BitNet weight quantization: quantize weights to {-1, 0, +1} using 
    absmean method.
    
    This is the core quantization method from BitNet, where weights are
    quantized to ternary values based on their absolute mean.
    
    Args:
        weights: Input weight tensor of any shape
        
    Returns:
        Quantized weights in {-1, 0, +1}
        
    Example:
        >>> weights = torch.randn(10, 10)
        >>> q_weights = absmean_quantize_weights(weights)
        >>> unique_vals = torch.unique(q_weights)
        >>> print(unique_vals)  # Should be close to [-1., 0., 1.]
    """
    # Calculate scaling factor (mean of absolute values)
    scale = torch.mean(torch.abs(weights))
    
    # Quantize to {-1, 0, +1}
    # If |w| > scale, keep sign; if |w| <= scale, set to 0
    quantized = torch.sign(weights) * (torch.abs(weights) > scale).float()
    
    return quantized


def absmax_quantize_activations(
    activations: torch.Tensor, 
    bits: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BitNet activation quantization: quantize activations to n-bit integers 
    using absmax method.
    
    Args:
        activations: Input activation tensor of any shape
        bits: Number of bits for quantization (default: 8)
        
    Returns:
        Tuple of (quantized_activations, scale_factor)
        
    Example:
        >>> acts = torch.randn(4, 256) 
        >>> q_acts, scale = absmax_quantize_activations(acts, bits=8)
        >>> print(f"Original range: [{acts.min():.3f}, {acts.max():.3f}]")
        >>> print(f"Quantized range: [{q_acts.min():.0f}, {q_acts.max():.0f}]")
    """
    # Calculate scaling factor (maximum absolute value)
    scale = torch.max(torch.abs(activations))
    
    # Handle edge case where all activations are zero
    if scale == 0:
        return torch.zeros_like(activations), torch.tensor(1.0)
    
    # Quantize to [-2^(bits-1), 2^(bits-1) - 1]
    max_val = 2**(bits-1) - 1
    quantized = torch.round(activations / scale * max_val)
    quantized = torch.clamp(quantized, -max_val-1, max_val)
    
    return quantized, scale


def dequantize_activations(
    quantized_activations: torch.Tensor, 
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize activations back to floating point.
    
    Args:
        quantized_activations: Quantized integer activations
        scale: Scale factor used during quantization
        
    Returns:
        Dequantized floating point activations
    """
    return quantized_activations * scale / 127.0  # Assuming 8-bit


def get_quantization_stats(
    original: torch.Tensor, 
    quantized: torch.Tensor
) -> dict:
    """
    Get statistics comparing original and quantized tensors.
    
    Args:
        original: Original tensor
        quantized: Quantized tensor
        
    Returns:
        Dictionary with comparison statistics
    """
    mse = torch.mean((original - quantized) ** 2).item()
    max_error = torch.max(torch.abs(original - quantized)).item()
    unique_values = torch.unique(quantized).numel()
    
    return {
        "mse": mse,
        "max_error": max_error,
        "unique_values": unique_values,
        # Rough estimate of compression ratio
        "compression_ratio": original.numel() * 32 / (unique_values * 2),
    } 