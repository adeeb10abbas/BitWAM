#!/usr/bin/env python3
"""
Basic Usage Example for 1-bit VLA

This script demonstrates how to use the core VLA BitNet model for 
basic vision-language-action inference.
"""

import torch
import matplotlib.pyplot as plt

# Import from our package
from bit_vla import (
    VLABitNet, 
    create_sample_data, 
    print_model_info,
    analyze_quantization
)


def main():
    """Demonstrate basic VLA BitNet usage."""
    print("🚀 1-bit VLA Basic Usage Example")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create VLA model
    print("\n📱 Creating VLA BitNet model...")
    model = VLABitNet(
        hidden_dim=512,
        action_dim=7,
        vocab_size=1000
    )
    model.to(device)
    model.eval()
    
    # Print model information
    print_model_info(model, "VLA BitNet")
    
    # Create sample data
    print("\n📊 Creating sample data...")
    data = create_sample_data(batch_size=4)
    
    # Move to device
    images = data["images"].to(device)
    tokens = data["token_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    
    print(f"  Images shape: {images.shape}")
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Attention mask shape: {attention_mask.shape}")
    
    # Forward pass
    print("\n🔄 Running inference...")
    with torch.no_grad():
        actions = model(images, tokens, attention_mask)
    
    print(f"  Predicted actions shape: {actions.shape}")
    print(f"  Sample action: {actions[0].cpu().numpy()}")
    
    # Analyze quantization
    print("\n🔍 Quantization Analysis:")
    print("-" * 30)
    
    # Run a forward pass to populate quantization statistics
    with torch.no_grad():
        _ = model(images[:2], tokens[:2], attention_mask[:2])
    
    # Get quantization summary
    summary = model.get_quantization_summary()
    
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  BitLinear parameters: {summary['bitlinear_parameters']:,}")
    print(f"  FP16 parameters: {summary['fp16_parameters']:,}")
    print(f"  Quantized ratio: {summary['quantized_ratio']:.1f}%")
    print(f"  FP32 size: {summary['estimated_size_mb']['fp32']:.1f} MB")
    print(f"  Quantized size: {summary['estimated_size_mb']['quantized']:.1f} MB")
    
    memory_savings = summary['memory_savings']
    print(f"  Memory saved: {memory_savings['absolute']:.1f} MB")
    print(f"  Memory reduction: {memory_savings['percentage']:.1f}%")
    
    # Test individual components
    print("\n🧩 Testing Individual Components:")
    print("-" * 30)
    
    # Vision only
    with torch.no_grad():
        visual_features = model.encode_vision(images)
    print(f"  Visual features shape: {visual_features.shape}")
    
    # Language only
    with torch.no_grad():
        language_features = model.encode_language(tokens, attention_mask)
    print(f"  Language features shape: {language_features.shape}")
    
    # Create visualization
    print("\n📈 Creating visualization...")
    create_analysis_plot(model, summary)
    
    print("\n✅ Basic usage demonstration completed!")
    print("See 'basic_usage_analysis.png' for visualizations.")


def create_analysis_plot(model, summary):
    """Create analysis plots for the model."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Parameter distribution
    bitlinear_params = summary['bitlinear_parameters']
    fp16_params = summary['fp16_parameters']
    
    labels = ['BitLinear\n(1.58-bit)', 'FP16 Layers']
    sizes = [bitlinear_params, fp16_params]
    colors = ['lightblue', 'lightcoral']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    ax1.set_title('Parameter Distribution')
    
    # 2. Memory comparison
    fp32_size = summary['estimated_size_mb']['fp32']
    quantized_size = summary['estimated_size_mb']['quantized']
    
    memory_data = [fp32_size, quantized_size]
    memory_labels = ['FP32', 'Quantized']
    
    bars = ax2.bar(memory_labels, memory_data, color=['red', 'green'])
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Memory Usage Comparison')
    
    # Add value labels on bars
    for bar, value in zip(bars, memory_data):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f} MB', ha='center', va='bottom')
    
    # 3. BitLinear layer analysis
    bitlinear_count = 0
    layer_names = []
    sparsity_values = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'quantize_weights'):
            bitlinear_count += 1
            layer_names.append(f"Layer {bitlinear_count}")
            
            # Get sparsity if quantized weights exist
            if hasattr(module, 'quantized_weight') and module.quantized_weight is not None:
                sparsity = (module.quantized_weight == 0).float().mean().item()
                sparsity_values.append(sparsity * 100)
            else:
                sparsity_values.append(0)
    
    if sparsity_values:
        ax3.bar(range(len(layer_names)), sparsity_values, color='skyblue')
        ax3.set_xlabel('BitLinear Layers')
        ax3.set_ylabel('Sparsity (%)')
        ax3.set_title('Sparsity by Layer')
        ax3.set_xticks(range(len(layer_names)))
        ax3.set_xticklabels(layer_names, rotation=45)
    
    # 4. Efficiency metrics
    metrics = ['Model Size', 'Memory Usage', 'Energy*']
    improvements = [
        quantized_size / fp32_size,
        quantized_size / fp32_size,
        0.3  # Estimated energy reduction
    ]
    
    colors_metrics = ['blue', 'green', 'orange']
    bars = ax4.bar(metrics, improvements, color=colors_metrics)
    ax4.set_ylabel('Relative to FP32')
    ax4.set_title('Efficiency Improvements')
    ax4.set_ylim(0, 1.2)
    
    # Add percentage labels
    for bar, improvement in zip(bars, improvements):
        reduction = (1 - improvement) * 100
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'-{reduction:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('basic_usage_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main() 