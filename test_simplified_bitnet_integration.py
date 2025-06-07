#!/usr/bin/env python3
"""
Simplified BitNet Integration Test

This script tests BitNet-style optimizations with the lerobot testing framework,
focusing on batch=1 performance for real-time robotics inference.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bit_vla.policies.bitact_policy import BitACTPolicy, BitACTConfig


class SimplifiedBitLinear(nn.Module):
    """
    Simplified BitLinear implementation that focuses on batch=1 optimization.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Global scale for simplified quantization
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simplified 1-bit quantization optimized for batch=1."""
        # Ultra-fast weight quantization: just sign
        w_quantized = torch.sign(self.weight) * self.scale
        
        # Simplified activation quantization for batch=1
        if x.shape[0] == 1:  # Batch=1 optimization
            x_abs_max = torch.max(torch.abs(x))
            if x_abs_max > 0:
                x_quantized = torch.sign(x) * x_abs_max
            else:
                x_quantized = x
        else:
            # Standard quantization for larger batches
            x_quantized = torch.sign(x) * torch.mean(torch.abs(x), dim=-1, keepdim=True)
        
        # Linear operation
        output = F.linear(x_quantized, w_quantized, self.bias)
        return output


class SimplifiedBitAttention(nn.Module):
    """
    Simplified multi-head attention optimized for BitNet and batch=1.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Use simplified BitLinear for projections
        self.q_proj = SimplifiedBitLinear(embed_dim, embed_dim)
        self.k_proj = SimplifiedBitLinear(embed_dim, embed_dim)
        self.v_proj = SimplifiedBitLinear(embed_dim, embed_dim)
        self.out_proj = SimplifiedBitLinear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim ** -0.5)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                is_causal: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Optimized attention for batch=1."""
        batch_size, seq_len, embed_dim = query.shape
        
        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Batch=1 optimized attention
        if batch_size == 1:
            # Simplified attention computation for single batch
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if is_causal:
                mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1)
                scores = scores.masked_fill(mask.bool(), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
        else:
            # Use standard scaled dot-product attention for larger batches
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal
            )
            attn_weights = None
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class SimplifiedBitFFN(nn.Module):
    """
    Simplified feed-forward network with BitLinear layers.
    """
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = SimplifiedBitLinear(dim, hidden_dim)
        self.linear2 = SimplifiedBitLinear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN."""
        x = self.linear1(x)
        x = F.relu(x)  # Using ReLU for simplicity
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class OptimizedBitNetACTPolicy(nn.Module):
    """
    BitACT Policy enhanced with simplified but optimized BitNet-style kernels.
    
    This version uses custom simplified implementations to avoid dependency issues
    while still demonstrating the performance benefits of BitNet-style optimizations.
    """
    
    def __init__(self, config: BitACTConfig, observation_dim: int = 8):
        super().__init__()
        self.config = config
        self.observation_dim = observation_dim
        
        # Feature extraction with simplified BitLinear layers
        self.feature_extractor = nn.Sequential(
            SimplifiedBitLinear(observation_dim, config.dim_model),
            nn.LayerNorm(config.dim_model),
            nn.ReLU(),
            SimplifiedBitLinear(config.dim_model, config.dim_model),
            nn.LayerNorm(config.dim_model),
            nn.ReLU(),
        )
        
        # Simplified BitNet attention layers
        self.attention_layers = nn.ModuleList([
            SimplifiedBitAttention(
                embed_dim=config.dim_model,
                num_heads=config.n_heads,
                dropout=0.1,
            )
            for _ in range(config.n_encoder_layers)
        ])
        
        # Simplified BitNet feed-forward layers
        self.ffn_layers = nn.ModuleList([
            SimplifiedBitFFN(
                dim=config.dim_model,
                hidden_dim=config.dim_feedforward,
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
        self.action_head = SimplifiedBitLinear(config.dim_model, config.action_dim)
        
        # VAE components (optional)
        if config.use_vae:
            self.mu_head = SimplifiedBitLinear(config.dim_model, config.latent_dim)
            self.log_sigma_x2_head = SimplifiedBitLinear(config.dim_model, config.latent_dim)
            self.latent_projection = nn.Linear(config.latent_dim, config.dim_model)
        
        print(f"🚀 OptimizedBitNetACT initialized:")
        print(f"   - Simplified BitNet attention layers: {len(self.attention_layers)}")
        print(f"   - Simplified BitNet FFN layers: {len(self.ffn_layers)}")
        print(f"   - Model dimension: {config.dim_model}")
        print(f"   - Attention heads: {config.n_heads}")
        print(f"   - Optimized for batch=1 inference")
    
    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations using simplified BitNet attention."""
        # Feature extraction
        x = self.feature_extractor(observations)  # [batch, dim_model]
        x = x.unsqueeze(1)  # [batch, 1, dim_model]
        
        # Apply simplified BitNet attention layers
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
        
        # Action prediction
        actions = self.action_head(decoder_input)
        
        return actions  # [batch, chunk_size, action_dim]
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through OptimizedBitNetACT policy."""
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


def test_simplified_bitnet_comparison():
    """Compare simplified BitNet kernels vs standard implementation for batch=1."""
    print("🚀 Simplified BitNet Integration Test (Batch=1 Focus)")
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
    
    # 2. Optimized BitNet kernels policy
    optimized_model = OptimizedBitNetACTPolicy(config, observation_dim=obs_dim)
    print("✅ Optimized BitNet policy created")
    
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
    optimized_params = count_parameters(optimized_model)
    fp32_params = count_parameters(fp32_model)
    
    print("📊 Model Statistics:")
    print(f"  - Standard BitACT: {standard_params:,} parameters")
    print(f"  - Optimized BitNet: {optimized_params:,} parameters")
    print(f"  - FP32 Baseline:   {fp32_params:,} parameters")
    print()
    
    # Benchmark each model
    models = {
        'FP32 Baseline': fp32_model,
        'Standard BitACT': standard_model,
        'Optimized BitNet': optimized_model,
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
    
    if results['FP32 Baseline'] and results['Optimized BitNet']:
        fp32_time = results['FP32 Baseline']['avg_inference_time_ms']
        optimized_time = results['Optimized BitNet']['avg_inference_time_ms']
        speedup = fp32_time / optimized_time
        
        print(f"FP32 Baseline:    {fp32_time:.3f} ms")
        print(f"Optimized BitNet: {optimized_time:.3f} ms")
        print(f"Speedup ratio:    {speedup:.2f}x")
        
        if speedup > 1.0:
            print("🎉 Optimized BitNet is FASTER!")
        else:
            print("⚠️  Optimized BitNet is slower")
    
    if results['Standard BitACT'] and results['Optimized BitNet']:
        standard_time = results['Standard BitACT']['avg_inference_time_ms']
        optimized_time = results['Optimized BitNet']['avg_inference_time_ms']
        improvement = standard_time / optimized_time
        
        print(f"\nStandard BitACT:  {standard_time:.3f} ms")
        print(f"Optimized BitNet: {optimized_time:.3f} ms")
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
    
    # Configure optimized BitNet policy for robotics
    config = BitACTConfig(
        dim_model=256,
        n_heads=8,
        n_encoder_layers=3,
        n_decoder_layers=3,
        chunk_size=8,
        action_dim=action_dim,
        use_bitnet=True,
        use_vae=False,
    )
    
    # Create optimized BitNet policy
    policy = OptimizedBitNetACTPolicy(config, observation_dim=obs_dim).to(device)
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


def test_batch_scalability():
    """Test how the optimizations scale with different batch sizes."""
    print("\n📈 Batch Size Scalability Test")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_dim = 32
    action_dim = 14
    
    config = BitACTConfig(
        dim_model=256,
        n_heads=8,
        n_encoder_layers=3,
        n_decoder_layers=3,
        chunk_size=8,
        action_dim=action_dim,
        use_bitnet=True,
        use_vae=False,
    )
    
    # Create optimized model
    model = OptimizedBitNetACTPolicy(config, observation_dim=obs_dim).to(device)
    model.eval()
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...")
        
        # Create input
        obs = torch.randn(batch_size, obs_dim, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(obs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        num_iterations = 100
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(obs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.perf_counter() - start_time
        avg_time = (total_time / num_iterations) * 1000
        throughput = (batch_size * num_iterations) / total_time
        
        results[batch_size] = {
            'avg_time_ms': avg_time,
            'throughput_samples_per_sec': throughput,
            'time_per_sample_ms': avg_time / batch_size,
        }
        
        print(f"  Avg time: {avg_time:.3f} ms")
        print(f"  Time per sample: {avg_time / batch_size:.3f} ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")
    
    print("\n📊 Batch Size Comparison:")
    print("Batch | Total Time | Per Sample | Throughput")
    print("------|------------|------------|------------")
    for batch_size, result in results.items():
        print(f"{batch_size:5d} | {result['avg_time_ms']:8.3f} ms | "
              f"{result['time_per_sample_ms']:8.3f} ms | "
              f"{result['throughput_samples_per_sec']:8.1f} /s")
    
    return results


if __name__ == "__main__":
    print("🚀 Simplified BitNet Integration Testing")
    print("=" * 60)
    
    # Test 1: Main comparison
    main_results = test_simplified_bitnet_comparison()
    
    # Test 2: LeRobot integration
    lerobot_results = test_lerobot_integration()
    
    # Test 3: Batch scalability
    scalability_results = test_batch_scalability()
    
    print("\n📋 Summary:")
    print("=" * 30)
    print("✅ Simplified BitNet integration completed")
    print("✅ LeRobot integration verified")
    print("✅ Batch scalability tested")
    
    # Save results
    results_summary = {
        'main_comparison': main_results,
        'lerobot_integration': lerobot_results,
        'batch_scalability': scalability_results,
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    import json
    output_file = "simplified_bitnet_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"📊 Results saved to: {output_file}") 