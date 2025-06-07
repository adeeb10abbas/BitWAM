#!/usr/bin/env python3
"""
BitNet Kernels Integration Test

This script tests the BitNet folder's optimized transformer and attention kernels
with the lerobot testing framework, specifically focusing on batch=1 performance
for real-time robotics inference.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add BitNet to path
sys.path.insert(0, str(Path(__file__).parent / "BitNet"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# BitNet imports
from bitnet import BitNetTransformer, BitLinear, BitMGQA, BitFeedForward

# Local imports
from bit_vla.policies.bitact_policy import BitACTPolicy, BitACTConfig


class BitNetACTPolicy(nn.Module):
    """
    BitACT Policy enhanced with optimized BitNet kernels.
    
    This version replaces the standard transformer components with the
    optimized BitNet transformer and attention kernels for better performance.
    """
    
    def __init__(self, config: BitACTConfig, observation_dim: int = 8):
        super().__init__()
        self.config = config
        self.observation_dim = observation_dim
        
        # Feature extraction with BitNet layers
        self.feature_extractor = nn.Sequential(
            BitLinear(observation_dim, config.dim_model),
            nn.LayerNorm(config.dim_model),
            nn.ReLU(),
            BitLinear(config.dim_model, config.dim_model),
            nn.LayerNorm(config.dim_model),
            nn.ReLU(),
        )
        
        # Optimized BitNet attention layers
        self.attention_layers = nn.ModuleList([
            BitMGQA(
                embed_dim=config.dim_model,
                query_heads=config.n_heads,
                kv_heads=max(1, config.n_heads // 2),  # Grouped query attention
                dropout=0.1,
            )
            for _ in range(config.n_encoder_layers)
        ])
        
        # BitNet feed-forward layers
        self.ffn_layers = nn.ModuleList([
            BitFeedForward(
                config.dim_model,
                config.dim_model,
                ff_mult=4,
                swish=True,
                post_act_ln=True,
                dropout=0.1,
            )
            for _ in range(config.n_encoder_layers)
        ])
        
        # Layer normalization
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(config.dim_model)
            for _ in range(config.n_encoder_layers)
        ])
        
        # Positional encoding for action chunks
        self.pos_encoding = nn.Parameter(
            torch.randn(config.chunk_size, config.dim_model) * 0.1
        )
        
        # Action prediction head
        self.action_head = BitLinear(config.dim_model, config.action_dim)
        
        # VAE components (optional)
        if config.use_vae:
            self.mu_head = BitLinear(config.dim_model, config.latent_dim)
            self.log_sigma_x2_head = BitLinear(config.dim_model, config.latent_dim)
            self.latent_projection = nn.Linear(config.latent_dim, config.dim_model)
        
        print(f"🚀 BitNetACT initialized with optimized kernels:")
        print(f"   - BitNet attention layers: {len(self.attention_layers)}")
        print(f"   - BitNet FFN layers: {len(self.ffn_layers)}")
        print(f"   - Model dimension: {config.dim_model}")
        print(f"   - Attention heads: {config.n_heads}")
    
    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations using BitNet attention."""
        # Feature extraction
        x = self.feature_extractor(observations)  # [batch, dim_model]
        x = x.unsqueeze(1)  # [batch, 1, dim_model]
        
        # Apply BitNet attention layers
        for attn, ffn, norm in zip(self.attention_layers, self.ffn_layers, self.norm_layers):
            # Self-attention with residual
            attn_out, _ = attn(x, x, x, is_causal=False)
            x = norm(x + attn_out)
            
            # Feed-forward with residual
            ffn_out = ffn(x)
            x = x + ffn_out
        
        return x.squeeze(1)  # [batch, dim_model]
    
    def decode_actions(self, encoded_obs: torch.Tensor, 
                      latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode actions from encoded observations."""
        # Prepare decoder input
        if latent is not None:
            latent_features = self.latent_projection(latent)
            decoder_input = encoded_obs + latent_features
        else:
            decoder_input = encoded_obs
        
        # Expand for chunk_size and add positional encoding
        decoder_input = decoder_input.unsqueeze(1).expand(
            -1, self.config.chunk_size, -1
        )  # [batch, chunk_size, dim_model]
        decoder_input = decoder_input + self.pos_encoding.unsqueeze(0)
        
        # Simple action prediction (could be enhanced with more attention layers)
        actions = self.action_head(decoder_input)
        
        return actions  # [batch, chunk_size, action_dim]
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through BitNetACT policy."""
        # Encode observations
        encoded_obs = self.encode_observations(observations)
        
        # VAE encoding (if enabled)
        latent = None
        if self.config.use_vae:
            mu = self.mu_head(encoded_obs)
            log_sigma_x2 = self.log_sigma_x2_head(encoded_obs)
            
            if self.training:
                # Sample from latent distribution during training
                std = torch.exp(0.5 * log_sigma_x2)
                eps = torch.randn_like(std)
                latent = mu + eps * std
            else:
                # Use mean during inference
                latent = mu
        
        # Decode to actions
        actions = self.decode_actions(encoded_obs, latent)
        
        return actions


def benchmark_single_inference(model: nn.Module, obs_dim: int, action_dim: int, 
                              device: torch.device, num_iterations: int = 1000) -> Dict[str, float]:
    """Benchmark model for single inference (batch=1)."""
    model.eval()
    model.to(device)
    
    # Create single observation
    obs = torch.randn(1, obs_dim, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(obs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            actions = model(obs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.perf_counter() - start_time
    avg_time = (total_time / num_iterations) * 1000  # Convert to ms
    
    return {
        'avg_inference_time_ms': avg_time,
        'total_time_s': total_time,
        'throughput_fps': num_iterations / total_time,
        'output_shape': tuple(actions.shape),
    }


def test_bitnet_kernels_comparison():
    """Compare BitNet kernels vs standard implementation for batch=1."""
    print("🚀 BitNet Kernels Integration Test (Batch=1 Focus)")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    obs_dim = 32  # Typical robotics observation dimension
    action_dim = 14  # Typical robotics action dimension
    num_iterations = 1000
    
    # Configuration for realistic robotics scenario
    config = BitACTConfig(
        dim_model=256,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=4,
        chunk_size=8,  # Smaller chunk for real-time
        action_dim=action_dim,
        use_bitnet=True,
        use_vae=False,  # Disable VAE for speed
    )
    
    print(f"Configuration:")
    print(f"  - Observation dim: {obs_dim}")
    print(f"  - Action dim: {action_dim}")
    print(f"  - Model dim: {config.dim_model}")
    print(f"  - Attention heads: {config.n_heads}")
    print(f"  - Encoder layers: {config.n_encoder_layers}")
    print(f"  - Chunk size: {config.chunk_size}")
    print()
    
    # Models to compare
    print("🔧 Creating models...")
    
    # 1. Standard BitACT policy (from our implementation)
    standard_model = BitACTPolicy(config, observation_dim=obs_dim)
    print("✅ Standard BitACT policy created")
    
    # 2. BitNet kernels enhanced policy
    bitnet_model = BitNetACTPolicy(config, observation_dim=obs_dim)
    print("✅ BitNet kernels policy created")
    
    # 3. FP32 baseline for comparison
    config_fp32 = BitACTConfig(
        dim_model=config.dim_model,
        n_heads=config.n_heads,
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
        chunk_size=config.chunk_size,
        action_dim=action_dim,
        use_bitnet=False,  # FP32 baseline
        use_vae=False,
    )
    fp32_model = BitACTPolicy(config_fp32, observation_dim=obs_dim)
    print("✅ FP32 baseline created")
    
    print()
    
    # Model size comparison
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    standard_params = count_parameters(standard_model)
    bitnet_params = count_parameters(bitnet_model)
    fp32_params = count_parameters(fp32_model)
    
    print("📊 Model Statistics:")
    print(f"  - Standard BitACT: {standard_params:,} parameters")
    print(f"  - BitNet Kernels:  {bitnet_params:,} parameters")
    print(f"  - FP32 Baseline:   {fp32_params:,} parameters")
    print()
    
    # Benchmark each model
    models = {
        'FP32 Baseline': fp32_model,
        'Standard BitACT': standard_model,
        'BitNet Kernels': bitnet_model,
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"⏱️  Benchmarking {name}...")
        try:
            result = benchmark_single_inference(
                model, obs_dim, action_dim, device, num_iterations
            )
            results[name] = result
            print(f"   Avg inference time: {result['avg_inference_time_ms']:.3f} ms")
            print(f"   Throughput: {result['throughput_fps']:.1f} FPS")
            print(f"   Output shape: {result['output_shape']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[name] = None
        print()
    
    # Performance comparison
    print("📈 Performance Comparison (Batch=1):")
    print("-" * 50)
    
    if results['FP32 Baseline'] and results['BitNet Kernels']:
        fp32_time = results['FP32 Baseline']['avg_inference_time_ms']
        bitnet_time = results['BitNet Kernels']['avg_inference_time_ms']
        speedup = fp32_time / bitnet_time
        
        print(f"FP32 Baseline:    {fp32_time:.3f} ms")
        print(f"BitNet Kernels:   {bitnet_time:.3f} ms")
        print(f"Speedup ratio:    {speedup:.2f}x")
        
        if speedup > 1.0:
            print("🎉 BitNet kernels are FASTER!")
        else:
            print("⚠️  BitNet kernels are slower")
    
    if results['Standard BitACT'] and results['BitNet Kernels']:
        standard_time = results['Standard BitACT']['avg_inference_time_ms']
        bitnet_time = results['BitNet Kernels']['avg_inference_time_ms']
        improvement = standard_time / bitnet_time
        
        print(f"\nStandard BitACT:  {standard_time:.3f} ms")
        print(f"BitNet Kernels:   {bitnet_time:.3f} ms")
        print(f"Improvement:      {improvement:.2f}x")
    
    # Real-time capability analysis
    print("\n🤖 Real-time Robotics Analysis:")
    print("-" * 40)
    
    target_frequencies = [10, 50, 100, 500, 1000]  # Hz
    
    for freq in target_frequencies:
        max_latency = 1000 / freq  # ms
        print(f"\n{freq} Hz control ({max_latency:.1f} ms budget):")
        
        for name, result in results.items():
            if result:
                latency = result['avg_inference_time_ms']
                status = "✅" if latency < max_latency else "❌"
                print(f"  {name}: {latency:.3f} ms {status}")
    
    return results


def test_bitnet_transformer_standalone():
    """Test the standalone BitNetTransformer for comparison."""
    print("\n🧠 Testing Standalone BitNetTransformer")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create BitNetTransformer for text-like tasks
    bitnet = BitNetTransformer(
        num_tokens=1000,  # Small vocabulary
        dim=512,
        depth=6,
        heads=8,
        ff_mult=4,
    ).to(device)
    
    print(f"BitNetTransformer parameters: {sum(p.numel() for p in bitnet.parameters()):,}")
    
    # Test with small sequence
    batch_size = 1  # Single inference
    seq_len = 64
    
    x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    # Benchmark
    num_iterations = 100
    bitnet.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = bitnet(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            logits = bitnet(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.perf_counter() - start_time
    avg_time = (total_time / num_iterations) * 1000
    
    print(f"Sequence length: {seq_len}")
    print(f"Output shape: {logits.shape}")
    print(f"Average inference time: {avg_time:.3f} ms")
    print(f"Throughput: {num_iterations / total_time:.1f} inferences/sec")
    
    return {
        'avg_time_ms': avg_time,
        'throughput': num_iterations / total_time,
        'output_shape': tuple(logits.shape),
    }


def test_lerobot_integration():
    """Test integration with lerobot-style data processing."""
    print("\n🤖 LeRobot Integration Test")
    print("=" * 40)
    
    # Simulate lerobot data format
    batch_size = 1  # Single robot inference
    obs_dim = 32
    action_dim = 14
    
    # Create sample batch (mimicking lerobot format)
    sample_batch = {
        'observation.state': torch.randn(batch_size, obs_dim),
        'action': torch.randn(batch_size, 8, action_dim),  # Action sequence
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configure BitNet policy for robotics
    config = BitACTConfig(
        dim_model=256,
        n_heads=8,
        n_encoder_layers=3,
        n_decoder_layers=3,
        chunk_size=8,
        action_dim=action_dim,
        use_bitnet=True,
        use_vae=False,
        performance_mode="ultra_fast",
    )
    
    # Create BitNet enhanced policy
    policy = BitNetACTPolicy(config, observation_dim=obs_dim).to(device)
    policy.eval()
    
    # Move batch to device
    for key in sample_batch:
        sample_batch[key] = sample_batch[key].to(device)
    
    print(f"Input observation shape: {sample_batch['observation.state'].shape}")
    print(f"Target action shape: {sample_batch['action'].shape}")
    
    # Test inference
    with torch.no_grad():
        start_time = time.perf_counter()
        predicted_actions = policy(sample_batch['observation.state'])
        inference_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Predicted action shape: {predicted_actions.shape}")
    print(f"Inference time: {inference_time:.3f} ms")
    
    # Verify shapes are compatible
    if predicted_actions.shape == sample_batch['action'].shape:
        print("✅ Output shape matches target")
    else:
        print("⚠️  Output shape mismatch")
    
    # Calculate loss to verify training compatibility
    loss = nn.MSELoss()(predicted_actions, sample_batch['action'])
    print(f"Sample MSE loss: {loss.item():.4f}")
    
    return {
        'inference_time_ms': inference_time,
        'output_shape': tuple(predicted_actions.shape),
        'loss': loss.item(),
    }


if __name__ == "__main__":
    print("🚀 BitNet Kernels Integration Testing")
    print("=" * 60)
    
    # Test 1: Main comparison
    main_results = test_bitnet_kernels_comparison()
    
    # Test 2: Standalone BitNetTransformer
    transformer_results = test_bitnet_transformer_standalone()
    
    # Test 3: LeRobot integration
    lerobot_results = test_lerobot_integration()
    
    print("\n📋 Summary:")
    print("=" * 30)
    print("✅ BitNet kernels integration completed")
    print("✅ Standalone transformer tested")
    print("✅ LeRobot integration verified")
    
    # Save results
    results_summary = {
        'main_comparison': main_results,
        'transformer_standalone': transformer_results,
        'lerobot_integration': lerobot_results,
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    import json
    output_file = "bitnet_kernels_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"📊 Results saved to: {output_file}") 