#!/usr/bin/env python3
"""
BitACT Policy Demo for 1-bit VLA Research

This script demonstrates the BitACT policy with action chunking and BitNet quantization.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from bit_vla import BitACTPolicy, BitACTConfig, print_model_info


def main():
    """Demonstrate BitACT policy functionality."""
    print("🎯 BitACT Policy Demonstration")
    print("=" * 50)
    
    # Create BitACT configuration
    config = BitACTConfig(
        action_dim=7,           # 7-DOF robot arm
        chunk_size=50,          # 50 action steps
        use_bitnet=True,
        dim_model=256,          # Smaller for demo
        n_encoder_layers=2,
        n_decoder_layers=3,
        use_vae=True,
        latent_dim=16
    )
    
    print("Configuration:")
    print(f"  Action dimension: {config.action_dim}")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  Model dimension: {config.dim_model}")
    print(f"  Using BitNet: {config.use_bitnet}")
    print(f"  Using VAE: {config.use_vae}")
    
    # Create policy
    print("\n🤖 Creating BitACT Policy...")
    policy = BitACTPolicy(config, observation_dim=8)
    policy.eval()
    
    # Print model information
    print_model_info(policy, "BitACT Policy")
    
    # Create sample observations (robot proprioception)
    batch_size = 4
    observations = torch.randn(batch_size, 8)
    
    print(f"\n📊 Input observations shape: {observations.shape}")
    
    # Forward pass - generate action chunks
    print("\n🔄 Generating action chunks...")
    with torch.no_grad():
        action_chunks = policy(observations)
    
    print(f"Action chunks shape: {action_chunks.shape}")
    print(f"Each observation produces {config.chunk_size} future actions")
    
    # Analyze the action chunks
    print("\n📈 Action Analysis:")
    print("-" * 30)
    
    # Statistics for first sample
    first_sample_actions = action_chunks[0].cpu().numpy()  # [chunk_size, action_dim]
    
    print(f"Action range: [{first_sample_actions.min():.3f}, {first_sample_actions.max():.3f}]")
    print(f"Action mean: {first_sample_actions.mean():.3f}")
    print(f"Action std: {first_sample_actions.std():.3f}")
    
    # Check action smoothness (important for robot control)
    action_diffs = np.diff(first_sample_actions, axis=0)
    smoothness = np.mean(np.abs(action_diffs))
    print(f"Action smoothness (lower is smoother): {smoothness:.3f}")
    
    # Get quantization summary
    summary = policy.get_quantization_summary()
    print(f"\n🔍 Quantization Summary:")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  BitLinear layers: {summary['bitlinear_layers']}")
    print(f"  Quantized ratio: {summary['quantized_ratio']:.1f}%")
    print(f"  Estimated memory: {summary['estimated_size_mb']['quantized']:.1f} MB")
    
    # Test parameter groups for BitNet optimization
    print("\n⚙️  BitNet Optimization Groups:")
    param_groups = policy.get_optim_params()
    for i, group in enumerate(param_groups):
        group_name = group.get("name", f"Group {i}")
        param_count = sum(p.numel() for p in group["params"])
        print(f"  {group_name}: {param_count:,} parameters")
    
    # Create visualization
    print("\n📊 Creating visualizations...")
    create_bitact_plots(action_chunks, config)
    
    # Test individual components
    print("\n🧩 Testing Individual Components:")
    test_components(policy, observations)
    
    print("\n✅ BitACT demonstration completed!")
    print("See 'bitact_analysis.png' for action visualizations.")


def create_bitact_plots(action_chunks, config):
    """Create plots showing BitACT action generation."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Action trajectory for first sample
    first_sample = action_chunks[0].cpu().numpy()
    time_steps = np.arange(config.chunk_size)
    
    for joint in range(min(3, config.action_dim)):  # Plot first 3 joints
        ax1.plot(time_steps, first_sample[:, joint], 
                label=f'Joint {joint+1}', alpha=0.8)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Action Value')
    ax1.set_title('Action Trajectory (First 3 Joints)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Action distribution across all samples
    all_actions = action_chunks.cpu().numpy().flatten()
    ax2.hist(all_actions, bins=50, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Action Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Action Value Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Action smoothness analysis
    batch_size, chunk_size, action_dim = action_chunks.shape
    smoothness_per_joint = []
    
    for joint in range(action_dim):
        joint_actions = action_chunks[:, :, joint].cpu().numpy()
        # Calculate smoothness for each sequence
        smoothness_values = []
        for seq in joint_actions:
            diffs = np.abs(np.diff(seq))
            smoothness_values.append(np.mean(diffs))
        smoothness_per_joint.append(np.mean(smoothness_values))
    
    joint_names = [f'Joint {i+1}' for i in range(action_dim)]
    bars = ax3.bar(joint_names, smoothness_per_joint, color='lightcoral')
    ax3.set_ylabel('Smoothness (lower = smoother)')
    ax3.set_title('Action Smoothness by Joint')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, smoothness_per_joint):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 4: Temporal correlation
    first_joint_actions = action_chunks[0, :, 0].cpu().numpy()
    correlations = []
    lags = range(1, min(11, len(first_joint_actions)))
    
    for lag in lags:
        corr = np.corrcoef(first_joint_actions[:-lag], first_joint_actions[lag:])[0, 1]
        correlations.append(corr)
    
    ax4.plot(lags, correlations, 'o-', color='green', alpha=0.8)
    ax4.set_xlabel('Lag (time steps)')
    ax4.set_ylabel('Correlation')
    ax4.set_title('Temporal Action Correlation')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('bitact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def test_components(policy, observations):
    """Test individual components of BitACT."""
    print("  Testing observation encoding...")
    
    with torch.no_grad():
        encoded_obs = policy.encode_observations(observations)
    print(f"    Encoded observations shape: {encoded_obs.shape}")
    
    if policy.config.use_vae:
        print("  Testing VAE components...")
        with torch.no_grad():
            mu = policy.mu_head(encoded_obs)
            log_sigma_x2 = policy.log_sigma_x2_head(encoded_obs)
        print(f"    Latent mu shape: {mu.shape}")
        print(f"    Latent log_sigma_x2 shape: {log_sigma_x2.shape}")
        
        # Sample from latent distribution
        std = torch.exp(0.5 * log_sigma_x2)
        eps = torch.randn_like(std)
        latent = mu + eps * std
        print(f"    Sampled latent shape: {latent.shape}")
        
        # Test action decoding with latent
        with torch.no_grad():
            actions_with_latent = policy.decode_actions(encoded_obs, latent)
        print(f"    Actions with latent shape: {actions_with_latent.shape}")
    
    print("  All components working correctly! ✓")


if __name__ == "__main__":
    main() 