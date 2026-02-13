#!/usr/bin/env python3
"""
Test and benchmark CUDA optimizations for BitNet models.

This script compares the performance of:
1. Standard BitLinear layers
2. CUDA-optimized BitLinear layers  
3. Full BitACT model optimizations
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent / "src"))

from bit_vla import BitACTPolicy, BitACTConfig
from bit_vla.models.bitlinear import BitLinear
from bit_vla.models.cuda_optimized_bitlinear import CudaOptimizedBitLinear, benchmark_cuda_optimizations


def test_individual_layer_performance():
    """Test performance of individual BitLinear layers."""
    print("🔬 Individual Layer Performance Test")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping GPU tests.")
        return
    
    device = torch.device('cuda')
    batch_sizes = [1, 8, 32, 64, 128]
    in_features = 512
    out_features = 256
    num_iterations = 1000
    
    print(f"Testing {num_iterations} iterations per configuration...")
    print()
    
    for batch_size in batch_sizes:
        print(f"📊 Batch size {batch_size}:")
        
        # Create test input
        x = torch.randn(batch_size, in_features, device=device)
        
        # Standard BitLinear
        standard_layer = BitLinear(in_features, out_features).to(device)
        standard_layer.eval()
        
        # CUDA-optimized BitLinear
        optimized_layer = CudaOptimizedBitLinear(in_features, out_features).to(device)
        optimized_layer.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = standard_layer(x)
                _ = optimized_layer(x)
        
        torch.cuda.synchronize()
        
        # Benchmark standard
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = standard_layer(x)
        torch.cuda.synchronize()
        standard_time = (time.perf_counter() - start_time) / num_iterations * 1000
        
        # Benchmark optimized
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = optimized_layer(x)
        torch.cuda.synchronize()
        optimized_time = (time.perf_counter() - start_time) / num_iterations * 1000
        
        speedup = standard_time / optimized_time
        throughput_standard = batch_size * 1000 / standard_time
        throughput_optimized = batch_size * 1000 / optimized_time
        
        print(f"  Standard:  {standard_time:.3f}ms ({throughput_standard:.0f} samples/s)")
        print(f"  Optimized: {optimized_time:.3f}ms ({throughput_optimized:.0f} samples/s)")
        print(f"  Speedup:   {speedup:.2f}x")
        print()


def test_full_model_performance():
    """Test performance of full BitACT models."""
    print("🤖 Full Model Performance Test") 
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping GPU tests.")
        return
    
    device = torch.device('cuda')
    obs_dim = 32
    action_dim = 14
    batch_sizes = [1, 8, 32]
    num_iterations = 100
    
    print(f"Testing {num_iterations} iterations per configuration...")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print()
    
    for batch_size in batch_sizes:
        print(f"📊 Batch size {batch_size}:")
        
        # Create test input
        x = torch.randn(batch_size, obs_dim, device=device)
        
        # Standard BitACT (no CUDA optimization)
        config_standard = BitACTConfig(
            action_dim=action_dim,
            use_bitnet=True,
            cuda_optimized=False
        )
        model_standard = BitACTPolicy(config_standard, observation_dim=obs_dim).to(device)
        model_standard.eval()
        
        # CUDA-optimized BitACT
        config_optimized = BitACTConfig(
            action_dim=action_dim,
            use_bitnet=True,
            cuda_optimized=True
        )
        model_optimized = BitACTPolicy(config_optimized, observation_dim=obs_dim).to(device)
        model_optimized.eval()
        
        # Apply inference optimizations
        model_optimized = model_optimized.optimize_for_inference()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model_standard(x)
                _ = model_optimized(x)
        
        torch.cuda.synchronize()
        
        # Benchmark standard
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model_standard(x)
        torch.cuda.synchronize()
        standard_time = (time.perf_counter() - start_time) / num_iterations * 1000
        
        # Benchmark optimized
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model_optimized(x)
        torch.cuda.synchronize()
        optimized_time = (time.perf_counter() - start_time) / num_iterations * 1000
        
        speedup = standard_time / optimized_time
        throughput_standard = batch_size * 1000 / standard_time
        throughput_optimized = batch_size * 1000 / optimized_time
        
        print(f"  Standard:  {standard_time:.3f}ms ({throughput_standard:.0f} samples/s)")
        print(f"  Optimized: {optimized_time:.3f}ms ({throughput_optimized:.0f} samples/s)")
        print(f"  Speedup:   {speedup:.2f}x")
        
        # Memory usage comparison
        memory_standard = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model_optimized(x)
        memory_optimized = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"  Memory:    {memory_standard:.1f}MB vs {memory_optimized:.1f}MB")
        print()


def test_vs_standard_fp32():
    """Compare optimized BitNet vs standard FP32 models."""
    print("⚡ BitNet vs FP32 Comparison (Optimized)")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping GPU tests.")
        return
    
    device = torch.device('cuda')
    obs_dim = 32
    action_dim = 14
    batch_sizes = [1, 8, 32, 64]
    num_iterations = 100
    
    for batch_size in batch_sizes:
        print(f"📊 Batch size {batch_size}:")
        
        # Create test input
        x = torch.randn(batch_size, obs_dim, device=device)
        
        # Standard FP32 BitACT
        config_fp32 = BitACTConfig(
            action_dim=action_dim,
            use_bitnet=False
        )
        model_fp32 = BitACTPolicy(config_fp32, observation_dim=obs_dim).to(device)
        model_fp32.eval()
        
        # Optimized BitNet BitACT
        config_bitnet = BitACTConfig(
            action_dim=action_dim,
            use_bitnet=True,
            cuda_optimized=True
        )
        model_bitnet = BitACTPolicy(config_bitnet, observation_dim=obs_dim).to(device)
        model_bitnet.eval()
        model_bitnet = model_bitnet.optimize_for_inference()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model_fp32(x)
                _ = model_bitnet(x)
        
        torch.cuda.synchronize()
        
        # Benchmark FP32
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model_fp32(x)
        torch.cuda.synchronize()
        fp32_time = (time.perf_counter() - start_time) / num_iterations * 1000
        
        # Benchmark BitNet
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model_bitnet(x)
        torch.cuda.synchronize()
        bitnet_time = (time.perf_counter() - start_time) / num_iterations * 1000
        
        speedup = fp32_time / bitnet_time
        throughput_fp32 = batch_size * 1000 / fp32_time
        throughput_bitnet = batch_size * 1000 / bitnet_time
        
        # Model size comparison
        fp32_params = sum(p.numel() * p.element_size() for p in model_fp32.parameters()) / 1024**2
        bitnet_params = sum(p.numel() * p.element_size() for p in model_bitnet.parameters()) / 1024**2
        
        print(f"  FP32:      {fp32_time:.3f}ms ({throughput_fp32:.0f} samples/s, {fp32_params:.1f}MB)")
        print(f"  BitNet:    {bitnet_time:.3f}ms ({throughput_bitnet:.0f} samples/s, {bitnet_params:.1f}MB)")
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"  Memory:    {fp32_params/bitnet_params:.2f}x reduction")
        print()


def run_optimization_recommendations():
    """Provide recommendations for optimal usage."""
    print("💡 Optimization Recommendations")
    print("=" * 50)
    
    recommendations = [
        "1. 🚀 Use CUDA-optimized BitLinear for GPU inference",
        "2. 📦 Call optimize_for_inference() before production use",
        "3. 🔄 Use larger batch sizes (32+) for better GPU utilization",
        "4. ⚡ Enable TensorFloat-32 for modern GPUs (automatic)",
        "5. 💾 BitNet provides memory savings with competitive speed",
        "6. 🎯 For real-time control, test on your specific hardware",
        "7. 📊 Profile your workload - results vary by model size",
    ]
    
    for rec in recommendations:
        print(rec)
    
    print()
    print("🎯 When to use BitNet:")
    print("  ✅ Memory-constrained environments")
    print("  ✅ Batch inference scenarios")
    print("  ✅ Model deployment at scale")
    print("  ⚠️  Consider FP32 for extremely latency-sensitive applications")


def main():
    print("🚀 CUDA BitNet Optimization Testing")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        print(f"🎮 GPU: {gpu_name}")
        print(f"🔥 CUDA Version: {torch.version.cuda}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("❌ CUDA not available - CPU tests only")
    
    print()
    
    # Run tests
    test_individual_layer_performance()
    test_full_model_performance()
    test_vs_standard_fp32()
    run_optimization_recommendations()
    
    print("✅ Testing complete!")


if __name__ == "__main__":
    main() 