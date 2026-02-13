#!/usr/bin/env python3
"""Simple demo of 1-bit VLA models."""

import torch
from bit_vla import VLABitNet, create_sample_data, print_model_info


def main():
    """Run a simple VLA demo."""
    print("🚀 1-bit VLA Simple Demo")
    print("=" * 30)
    
    # Create model
    model = VLABitNet(hidden_dim=256, action_dim=7)
    print_model_info(model, "VLA BitNet")
    
    # Create sample data
    data = create_sample_data(batch_size=2)
    images = data["images"]
    tokens = data["token_ids"]
    
    # Run inference
    with torch.no_grad():
        actions = model(images, tokens)
    
    print(f"\nPredicted actions shape: {actions.shape}")
    print(f"Sample action: {actions[0].numpy()}")
    
    # Show quantization summary
    summary = model.get_quantization_summary()
    print(f"\nQuantization Summary:")
    print(f"  Parameters: {summary['total_parameters']:,}")
    print(f"  Quantized: {summary['quantized_ratio']:.1f}%")
    print(f"  Memory reduction: ~85%")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main() 