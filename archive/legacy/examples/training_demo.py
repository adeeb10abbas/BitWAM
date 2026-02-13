#!/usr/bin/env python3
"""
Training Demo for 1-bit VLA Research

This script demonstrates how to train VLA models with BitNet quantization
using the provided training utilities.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from bit_vla import (
    VLABitNet, 
    VLATrainer,
    BitNetOptimizer,
    create_sample_data,
    print_model_info
)


def create_synthetic_dataset(num_samples: int = 1000) -> DataLoader:
    """Create a synthetic dataset for training demonstration."""
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    # Generate synthetic data
    all_images = []
    all_tokens = []
    all_masks = []
    all_actions = []
    
    for _ in range(num_samples):
        data = create_sample_data(batch_size=1)
        all_images.append(data["images"])
        all_tokens.append(data["token_ids"])
        all_masks.append(data["attention_mask"])
        all_actions.append(data["actions"])
    
    # Concatenate all data
    images = torch.cat(all_images, dim=0)
    tokens = torch.cat(all_tokens, dim=0)
    masks = torch.cat(all_masks, dim=0)
    actions = torch.cat(all_actions, dim=0)
    
    print(f"  Images: {images.shape}")
    print(f"  Tokens: {tokens.shape}")
    print(f"  Actions: {actions.shape}")
    
    # Create dataset and dataloader
    dataset = TensorDataset(images, tokens, masks, actions)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    
    return dataloader


def main():
    """Main training demonstration."""
    print("🎯 VLA Training Demonstration")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("\n📱 Creating VLA BitNet model...")
    model = VLABitNet(
        hidden_dim=256,    # Smaller for demo
        action_dim=7,
        vocab_size=1000
    )
    
    print_model_info(model, "VLA BitNet")
    
    # Create datasets
    print("\n📊 Creating training data...")
    train_loader = create_synthetic_dataset(num_samples=800)
    val_loader = create_synthetic_dataset(num_samples=200)
    
    # Create trainer
    print("\n🚀 Setting up trainer...")
    trainer = VLATrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        output_dir="outputs/training_demo",
        log_freq=50,
        save_freq=200,
        eval_freq=100
    )
    
    # Setup optimizer and loss
    trainer.setup_optimizer(
        stage1_lr=1e-3,
        stage2_lr=1e-4,
        stage1_steps=300,  # Reduced for demo
        weight_decay=0.1
    )
    trainer.setup_loss_function()
    
    # Demonstrate batch processing
    print("\n🔄 Testing batch processing...")
    sample_batch = next(iter(train_loader))
    
    # Convert to dict format expected by trainer
    batch_dict = {
        "images": sample_batch[0],
        "token_ids": sample_batch[1],
        "attention_mask": sample_batch[2],
        "actions": sample_batch[3]
    }
    
    print(f"Sample batch shapes:")
    for key, value in batch_dict.items():
        print(f"  {key}: {value.shape}")
    
    # Test single training step
    print("\n⚡ Testing single training step...")
    step_metrics = trainer.train_step(batch_dict)
    print(f"Step metrics: {step_metrics}")
    
    # Run short training
    print("\n🏃 Running training (500 steps)...")
    trainer.train(num_steps=500)
    
    # Create training analysis plots
    print("\n📈 Creating training analysis...")
    create_training_plots(trainer)
    
    # Test model after training
    print("\n🧪 Testing trained model...")
    test_trained_model(model, device)
    
    print("\n✅ Training demonstration completed!")
    print("See 'training_analysis.png' for training curves.")


def create_training_plots(trainer):
    """Create plots showing training progress."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training loss
    ax1.plot(trainer.train_losses, alpha=0.7, label='Train Loss')
    if trainer.val_losses:
        val_steps = [i * trainer.eval_freq for i in range(len(trainer.val_losses))]
        ax1.plot(val_steps, trainer.val_losses, 'r-', alpha=0.8, label='Val Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate schedule
    if trainer.lr_history:
        bitnet_lrs = [lr_info.get('bitnet_layers_lr', 0) for lr_info in trainer.lr_history]
        fp16_lrs = [lr_info.get('fp16_layers_lr', 0) for lr_info in trainer.lr_history]
        
        ax2.plot(bitnet_lrs, label='BitNet Layers', alpha=0.8)
        ax2.plot(fp16_lrs, label='FP16 Layers', alpha=0.8)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    # Loss smoothing
    if len(trainer.train_losses) > 50:
        # Simple moving average
        window = 20
        smoothed_loss = []
        for i in range(window, len(trainer.train_losses)):
            smoothed_loss.append(sum(trainer.train_losses[i-window:i]) / window)
        
        ax3.plot(range(window, len(trainer.train_losses)), smoothed_loss, 'g-', alpha=0.8)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Smoothed Loss')
        ax3.set_title('Loss Trend (Moving Average)')
        ax3.grid(True, alpha=0.3)
    
    # Model quantization summary
    summary = trainer.model.get_quantization_summary()
    
    # Memory usage pie chart
    fp32_size = summary['estimated_size_mb']['fp32']
    quantized_size = summary['estimated_size_mb']['quantized']
    
    labels = ['Quantized\nModel', 'Memory\nSaved']
    sizes = [quantized_size, fp32_size - quantized_size]
    colors = ['lightblue', 'lightcoral']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    ax4.set_title('Memory Usage\n(BitNet Quantization)')
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def test_trained_model(model, device):
    """Test the trained model on some sample data."""
    model.eval()
    
    # Create test data
    test_data = create_sample_data(batch_size=3)
    images = test_data["images"].to(device)
    tokens = test_data["token_ids"].to(device)
    masks = test_data["attention_mask"].to(device)
    
    print("Test data shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Tokens: {tokens.shape}")
    
    # Run inference
    with torch.no_grad():
        actions = model(images, tokens, masks)
    
    print(f"Predicted actions: {actions.shape}")
    print(f"Sample prediction: {actions[0].cpu().numpy()}")
    
    # Analyze quantization in trained model
    summary = model.get_quantization_summary()
    print(f"\nTrained model summary:")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Quantized ratio: {summary['quantized_ratio']:.1f}%")
    print(f"  Memory reduction: {summary['memory_savings']['percentage']:.1f}%")


if __name__ == "__main__":
    main() 