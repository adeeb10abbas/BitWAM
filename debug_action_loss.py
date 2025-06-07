#!/usr/bin/env python3
"""
Debug script to investigate why action loss is extremely high in BitACT training.

This will help us understand:
1. What range/scale the action data is in
2. What the model is actually predicting
3. Whether there's a shape or scale mismatch
4. If we need different loss computation or normalization
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project paths
sys.path.append(str(Path(__file__).parent / "src"))

from bit_vla import BitACTPolicy, BitACTConfig


def analyze_typical_robotics_actions():
    """Analyze typical robotics action ranges to understand the scale problem."""
    print(f"🔍 Analyzing Typical Robotics Action Ranges")
    
    # ALOHA dataset typically has 14 dimensions (7 per arm)
    # Each dimension represents joint positions/velocities
    
    # Simulate typical ALOHA action ranges based on joint limits
    print("\n📊 Expected Action Ranges for ALOHA:")
    
    # Joint position ranges (radians) - typical for robot arms
    joint_ranges = {
        "shoulder_pan": (-3.14, 3.14),     # ±180 degrees
        "shoulder_lift": (-1.57, 1.57),   # ±90 degrees  
        "elbow": (-3.14, 3.14),           # ±180 degrees
        "wrist_1": (-3.14, 3.14),         # ±180 degrees
        "wrist_2": (-3.14, 3.14),         # ±180 degrees
        "wrist_3": (-3.14, 3.14),         # ±180 degrees
        "gripper": (0.0, 1.0),            # 0=closed, 1=open
    }
    
    print("Per-joint expected ranges:")
    for joint, (min_val, max_val) in joint_ranges.items():
        print(f"  {joint}: [{min_val:.2f}, {max_val:.2f}] (range: {max_val-min_val:.2f})")
    
    # Expected action statistics
    expected_range = 6.28  # ±π radians
    expected_std = 1.0     # Typical std for normalized joint positions
    
    print(f"\nExpected action statistics:")
    print(f"  Typical range: ±{expected_range/2:.2f} (total: {expected_range:.2f})")
    print(f"  Expected std: ~{expected_std:.2f}")
    print(f"  Expected max loss (untrained): ~{expected_range**2:.1f}")
    
    return expected_range, expected_std


def test_loss_scales():
    """Test different action scales to understand the loss magnitude."""
    print(f"\n🧮 Testing Loss Scales for Different Action Ranges")
    
    batch_size = 16
    chunk_size = 50
    action_dim = 14
    
    # Create model
    config = BitACTConfig(
        action_dim=action_dim,
        chunk_size=chunk_size,
        use_vae=True,
        kl_weight=1.0,
    )
    
    model = BitACTPolicy(config, observation_dim=32)
    model.eval()
    
    # Test different action scales
    test_scales = [
        ("Small scale (normalized)", 1.0),
        ("Medium scale (radians)", 3.14),
        ("Large scale (degrees)", 180.0),
        ("Very large scale", 1000.0),
    ]
    
    observations = torch.randn(batch_size, 32)
    
    print("\nLoss analysis for different action scales:")
    print("Scale                     | Action Range | MSE Loss    | Action Std")
    print("-" * 70)
    
    for scale_name, scale_factor in test_scales:
        # Create actions at this scale
        base_actions = torch.randn(batch_size, chunk_size, action_dim) * scale_factor
        
        # Get model predictions (random at this point)
        with torch.no_grad():
            predicted_actions = model(observations)
        
        # Compute loss
        mse_loss = nn.MSELoss()(predicted_actions, base_actions)
        action_std = base_actions.std().item()
        action_range = (base_actions.max() - base_actions.min()).item()
        
        print(f"{scale_name:<25} | {action_range:>10.2f} | {mse_loss.item():>10.2f} | {action_std:>9.2f}")
    
    return test_scales


def simulate_training_scenario():
    """Simulate the training scenario with high loss values."""
    print(f"\n🎯 Simulating Your Training Scenario")
    
    # Based on your logs: action_loss=9207, kl_loss=0.087
    observed_action_loss = 9207.5986
    observed_kl_loss = 0.0873
    
    print(f"Observed losses from your training:")
    print(f"  Action Loss: {observed_action_loss:.4f}")
    print(f"  KL Loss: {observed_kl_loss:.4f}")
    print(f"  Total Loss: {observed_action_loss + observed_kl_loss:.4f}")
    
    # Reverse engineer what action scale would cause this loss
    # MSE loss ≈ (prediction - target)²
    # If predictions are ~0 (untrained) and targets are large, loss ≈ target²
    
    estimated_action_magnitude = np.sqrt(observed_action_loss)
    print(f"\nEstimated action magnitude: √{observed_action_loss:.0f} ≈ {estimated_action_magnitude:.1f}")
    
    # This suggests actions are in a very large scale!
    if estimated_action_magnitude > 50:
        print("🔥 CRITICAL: Actions are definitely NOT normalized!")
        print("   → Actions might be in degrees instead of radians")
        print("   → Or actions might have very large absolute values")
    elif estimated_action_magnitude > 10:
        print("⚠️  Actions seem to have large scale - normalization recommended")
    else:
        print("✅ Action scale seems reasonable - issue might be elsewhere")
    
    # Test different scenarios
    print(f"\nTesting scenarios that could cause loss = {observed_action_loss:.0f}:")
    
    scenarios = [
        ("Actions in degrees (0-360)", 180.0),
        ("Actions in large joint positions", 50.0),
        ("Actions with wrong scaling", 100.0),
        ("Normal actions (should be low loss)", 3.14),
    ]
    
    for scenario, scale in scenarios:
        # Simulate untrained model (predictions ≈ 0)
        predictions = torch.randn(1, 50, 14) * 0.1  # Small random predictions
        targets = torch.randn(1, 50, 14) * scale    # Actions at different scales
        
        loss = nn.MSELoss()(predictions, targets).item()
        print(f"  {scenario:<35}: Loss = {loss:>8.1f}")
        
        if abs(loss - observed_action_loss) < 1000:
            print(f"    ✅ MATCH! This could be the issue!")


def suggest_immediate_fixes():
    """Suggest immediate fixes for the high loss issue."""
    print(f"\n💡 Immediate Fixes to Try:")
    
    fixes = [
        "1. 🔥 Add action normalization to your dataset loading",
        "2. 📊 Check if actions are in degrees - convert to radians", 
        "3. 🎯 Use action statistics to normalize: (action - mean) / std",
        "4. 🛠️  Clip extreme action values during training",
        "5. ⚖️  Use Huber loss instead of MSE for robustness",
        "6. 📈 Scale the loss by action dimension (divide by action_dim)",
    ]
    
    for fix in fixes:
        print(fix)
    
    print(f"\n🚀 Quick Test - Add this to your training script:")
    print("""
# Before computing loss, normalize actions:
def normalize_actions(actions):
    # Clip extreme values
    actions = torch.clamp(actions, -100, 100)
    
    # Normalize to [-1, 1] range
    action_min = actions.min()
    action_max = actions.max()
    normalized = 2 * (actions - action_min) / (action_max - action_min) - 1
    return normalized

# In your training loop:
normalized_predicted = normalize_actions(predicted_actions)
normalized_targets = normalize_actions(actions)
action_loss = nn.MSELoss()(normalized_predicted, normalized_targets)
""")


def create_loss_analysis_plot():
    """Create a plot showing loss vs action scale relationship."""
    print(f"\n📈 Creating Loss vs Action Scale Analysis...")
    
    scales = np.logspace(-1, 3, 50)  # 0.1 to 1000
    losses = scales ** 2  # MSE loss ≈ scale²
    
    plt.figure(figsize=(12, 8))
    
    # Main loss curve
    plt.subplot(2, 2, 1)
    plt.loglog(scales, losses, 'b-', linewidth=2, label='Expected MSE Loss')
    plt.axhline(y=9207, color='r', linestyle='--', alpha=0.7, label='Your Observed Loss')
    plt.axvline(x=95.9, color='r', linestyle='--', alpha=0.7, label='Implied Action Scale')
    plt.xlabel('Action Scale (magnitude)')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss vs Action Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Action scale examples
    plt.subplot(2, 2, 2)
    scale_examples = [
        ("Normalized\n[-1, 1]", 1, 'green'),
        ("Radians\n[-π, π]", 3.14, 'blue'),  
        ("Degrees\n[-180, 180]", 180, 'orange'),
        ("Your Data\n(estimated)", 95.9, 'red'),
    ]
    
    names, values, colors = zip(*scale_examples)
    bars = plt.bar(names, values, color=colors, alpha=0.7)
    plt.ylabel('Action Scale')
    plt.title('Typical Action Scales')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{value:.1f}', ha='center', va='bottom')
    
    # Loss comparison
    plt.subplot(2, 2, 3)
    loss_examples = [s**2 for _, s, _ in scale_examples]
    bars = plt.bar(names, loss_examples, color=colors, alpha=0.7)
    plt.ylabel('Expected MSE Loss')
    plt.title('Expected Loss for Each Scale')
    plt.xticks(rotation=45)
    plt.yscale('log')
    
    # Add value labels
    for bar, loss in zip(bars, loss_examples):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5, 
                f'{loss:.0f}', ha='center', va='bottom')
    
    # Normalization benefit
    plt.subplot(2, 2, 4)
    before_loss = 9207
    after_loss = 1.0  # Normalized loss
    improvement = before_loss / after_loss
    
    plt.bar(['Before\nNormalization', 'After\nNormalization'], 
            [before_loss, after_loss], 
            color=['red', 'green'], alpha=0.7)
    plt.ylabel('MSE Loss')
    plt.title('Normalization Impact')
    plt.yscale('log')
    
    # Add improvement text
    plt.text(0.5, 100, f'{improvement:.0f}x\nImprovement!', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('loss_scale_analysis.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved loss analysis plot as 'loss_scale_analysis.png'")


def main():
    print("🔍 BitACT Action Loss Debugging - Simplified Version")
    print("="*60)
    
    # Analyze typical ranges
    analyze_typical_robotics_actions()
    
    # Test loss scales
    test_loss_scales()
    
    # Simulate your scenario
    simulate_training_scenario()
    
    # Create analysis plot
    create_loss_analysis_plot()
    
    # Suggest fixes
    suggest_immediate_fixes()
    
    print(f"\n🎯 CONCLUSION:")
    print("Your action loss of ~9207 strongly suggests that:")
    print("1. 🔥 Actions are NOT normalized (likely in degrees or large units)")
    print("2. 📊 Action magnitudes are around ±96 instead of ±1-3")
    print("3. 🛠️  You NEED action normalization before training")
    print("4. ⚡ This will likely fix the loss stagnation issue!")


if __name__ == "__main__":
    main() 