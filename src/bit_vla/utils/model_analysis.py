"""
Model analysis utilities for 1-bit VLA research.

Functions for analyzing model size, quantization statistics, and performance.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


def print_model_info(model: nn.Module, name: str = "Model") -> None:
    """
    Print detailed model information including quantization analysis.
    
    Args:
        model: PyTorch model to analyze
        name: Name to display for the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    print(f"\n🤖 {name} Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB (FP32)")
    
    # Analyze BitLinear layers
    bitlinear_count = 0
    bitlinear_params = 0
    fp16_params = 0
    
    for module_name, module in model.named_modules():
        if hasattr(module, 'quantize_weights'):  # BitLinear layer
            bitlinear_count += 1
            bitlinear_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, torch.nn.Linear):
            fp16_params += sum(p.numel() for p in module.parameters())
    
    print(f"  BitLinear layers: {bitlinear_count}")
    print(f"  BitLinear parameters: {bitlinear_params:,}")
    print(f"  FP16 parameters: {fp16_params:,}")
    
    if bitlinear_params > 0:
        quantized_ratio = bitlinear_params / total_params * 100
        print(f"  Quantized ratio: {quantized_ratio:.1f}%")
        
        # Estimate memory savings (1.58-bit ≈ 85% reduction)
        estimated_size = (
            bitlinear_params * 0.2 + fp16_params * 4
        ) / 1024**2  # MB
        print(f"  Estimated quantized size: {estimated_size:.1f} MB")


def analyze_quantization(
    model: nn.Module, 
    step: int, 
    loss_history: list
) -> Dict[str, float]:
    """
    Analyze quantization statistics during training.
    
    Args:
        model: Model to analyze
        step: Current training step
        loss_history: List of loss values
        
    Returns:
        Dictionary with quantization statistics
    """
    weight_scales = []
    activation_scales = []
    sparsity_ratios = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'quantize_weights'):
            # Get current weights and quantize them
            with torch.no_grad():
                weights = module.weight.data
                w_quantized, w_scale = module.quantize_weights(weights)
                
                # Calculate sparsity (percentage of zeros)
                sparsity = (w_quantized == 0).float().mean().item()
                sparsity_ratios.append(sparsity)
                
                # Store scales
                weight_scales.append(w_scale.mean().item())
                if (hasattr(module, 'activation_scale') and 
                        module.activation_scale is not None):
                    activation_scales.append(
                        module.activation_scale.item()
                    )
    
    stats = {}
    if weight_scales:
        avg_weight_scale = sum(weight_scales) / len(weight_scales)
        avg_activation_scale = (
            sum(activation_scales) / len(activation_scales) 
            if activation_scales else 0
        )
        avg_sparsity = sum(sparsity_ratios) / len(sparsity_ratios)
        
        stats = {
            "step": step,
            "loss": loss_history[-1] if loss_history else 0.0,
            "avg_weight_scale": avg_weight_scale,
            "avg_activation_scale": avg_activation_scale,
            "avg_sparsity": avg_sparsity,
        }
        
        print(f"  Step {step:4d} | Loss: {loss_history[-1]:.4f} | "
              f"Weight Scale: {avg_weight_scale:.4f} | "
              f"Act Scale: {avg_activation_scale:.4f} | "
              f"Sparsity: {avg_sparsity:.2%}")
    
    return stats


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    # 4 bytes per float32 parameter
    return total_params * 4 / 1024**2


def compare_models(
    model1: nn.Module, 
    model2: nn.Module, 
    names: tuple[str, str] = ("Model 1", "Model 2")
) -> Dict[str, Any]:
    """
    Compare two models in terms of size and parameters.
    
    Args:
        model1: First model to compare
        model2: Second model to compare  
        names: Names for the models
        
    Returns:
        Dictionary with comparison statistics
    """
    size1 = get_model_size_mb(model1)
    size2 = get_model_size_mb(model2)
    
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    
    comparison = {
        names[0]: {
            "parameters": params1,
            "size_mb": size1,
        },
        names[1]: {
            "parameters": params2,
            "size_mb": size2,
        },
        "comparison": {
            "size_reduction": (
                (1 - size2 / size1) * 100 if size1 > 0 else 0
            ),
            "param_reduction": (
                (1 - params2 / params1) * 100 if params1 > 0 else 0
            ),
        }
    }
    
    return comparison


def get_layer_wise_analysis(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Get layer-wise analysis of the model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with per-layer statistics
    """
    analysis = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            layer_info = {
                "type": type(module).__name__,
                "parameters": module.weight.numel(),
                "shape": list(module.weight.shape),
            }
            
            # Add quantization info if available
            if hasattr(module, 'get_quantization_stats'):
                layer_info["quantization"] = module.get_quantization_stats()
            
            analysis[name] = layer_info
    
    return analysis 