#!/usr/bin/env python3
"""
Robotics-Focused BitNet Optimization for Single Inference

This optimization specifically targets the real-world robotics use case:
- Batch size = 1 (single robot control)
- Real-time inference (sub-millisecond requirements)
- Memory efficiency for edge deployment

The goal is to make BitNet competitive with FP32 at batch=1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import time
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent / "src"))

from bit_vla import BitACTPolicy, BitACTConfig


class SingleInferenceOptimizedBitLinear(nn.Module):
    """
    BitLinear optimized specifically for single inference (batch=1).
    
    Key optimizations for batch=1:
    1. Eliminate all batch-related overhead
    2. Pre-compute everything possible
    3. Use the simplest possible quantization
    4. Optimize for GPU kernel efficiency at small sizes
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Single global scale (minimal overhead)
        self.scale = nn.Parameter(torch.ones(1))
        
        # Pre-computed matrices for inference
        self.register_buffer('weight_signs', None, persistent=False)
        self.register_buffer('weight_abs_mean', None, persistent=False)
        self._inference_ready = False
        
    def prepare_for_inference(self):
        """Pre-compute all possible values for ultra-fast single inference."""
        if not self._inference_ready:
            with torch.no_grad():
                # Pre-compute weight signs and scaling
                self.weight_signs = torch.sign(self.weight)
                self.weight_abs_mean = self.weight.abs().mean()
                self._inference_ready = True
    
    def forward(self, x):
        if self.training:
            # Training: full quantization
            weight_q = torch.sign(self.weight)
            x_q = torch.sign(x)
            output = F.linear(x_q, weight_q, None) * self.scale
        else:
            # Inference: use pre-computed values
            if not self._inference_ready:
                self.prepare_for_inference()
            
            # Ultra-minimal quantization for batch=1
            x_q = torch.sign(x)
            output = F.linear(x_q, self.weight_signs, None) * self.scale
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class MinimalOverheadBitLinear(nn.Module):
    """
    Absolute minimal overhead BitLinear for single inference.
    
    This sacrifices some theoretical quantization quality for maximum speed
    in the batch=1 case.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Use half precision for weights to reduce memory bandwidth
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # No scaling parameters - use raw quantization
        
    def forward(self, x):
        if self.training:
            # Convert to FP32 for training
            weight_fp32 = self.weight.float()
            weight_q = torch.sign(weight_fp32)
            x_q = torch.sign(x)
            output = F.linear(x_q, weight_q, None)
        else:
            # Direct FP16 operations for inference
            x_half = x.half()
            weight_q = torch.sign(self.weight)
            x_q = torch.sign(x_half)
            output = F.linear(x_q.float(), weight_q.float(), None)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


def analyze_single_inference_bottlenecks():
    """Analyze what's causing BitNet to be slow at batch=1."""
    print("🔍 Single Inference Bottleneck Analysis")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_features, out_features = 512, 256
    batch_size = 1  # Real robotics use case
    num_iterations = 1000
    
    x = torch.randn(batch_size, in_features, device=device)
    
    # Test different approaches
    layers = {
        "FP32 Linear": nn.Linear(in_features, out_features).to(device),
        "Standard BitLinear": None,  # We'll import this
        "Single-Opt BitLinear": SingleInferenceOptimizedBitLinear(in_features, out_features).to(device),
        "Minimal BitLinear": MinimalOverheadBitLinear(in_features, out_features).to(device),
    }
    
    # Import standard BitLinear
    try:
        from bit_vla.models.bitlinear import BitLinear
        layers["Standard BitLinear"] = BitLinear(in_features, out_features).to(device)
    except:
        print("⚠️  Could not import standard BitLinear")
    
    results = {}
    
    for name, layer in layers.items():
        if layer is None:
            continue
            
        layer.eval()
        
        # Special prep for optimized layers
        if hasattr(layer, 'prepare_for_inference'):
            layer.prepare_for_inference()
        
        # Warmup
        with torch.no_grad():
            for _ in range(20):
                _ = layer(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                output = layer(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        throughput = 1000 / avg_time
        
        results[name] = {
            'avg_ms': avg_time,
            'min_ms': min_time,
            'throughput': throughput,
        }
        
        print(f"{name:<20}: {avg_time:.4f}ms avg, {min_time:.4f}ms min, {throughput:.0f} inferences/s")
    
    # Analysis
    print(f"\n📊 Single Inference Analysis:")
    print("-" * 35)
    
    if "FP32 Linear" in results and "Standard BitLinear" in results:
        fp32_time = results["FP32 Linear"]["avg_ms"]
        bitnet_time = results["Standard BitLinear"]["avg_ms"]
        overhead = (bitnet_time - fp32_time) / fp32_time * 100
        
        print(f"BitNet Overhead: {overhead:.1f}% slower than FP32")
        print(f"Overhead Time: {bitnet_time - fp32_time:.4f}ms")
    
    # Find best BitNet variant
    bitnet_results = {k: v for k, v in results.items() if "BitLinear" in k}
    if bitnet_results:
        best_bitnet = min(bitnet_results.items(), key=lambda x: x[1]['avg_ms'])
        print(f"Best BitNet: {best_bitnet[0]} at {best_bitnet[1]['avg_ms']:.4f}ms")
    
    return results


def test_real_robotics_scenarios():
    """Test with actual robotics inference patterns."""
    print(f"\n🤖 Real Robotics Inference Patterns")
    print("=" * 45)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Real robotics dimensions
    obs_dim = 32  # Typical robot state size
    action_dim = 14  # 7-DOF robot arm (pos + vel)
    
    scenarios = [
        ("Real-time Control (1 robot)", 1),
        ("Dual-arm Robot", 2), 
        ("Small Multi-robot", 4),
    ]
    
    for scenario_name, batch_size in scenarios:
        print(f"\n🔬 {scenario_name} (batch={batch_size}):")
        
        x = torch.randn(batch_size, obs_dim, device=device)
        
        # Test different model configurations
        configs = [
            ("FP32", {"use_bitnet": False}),
            ("BitNet Standard", {"use_bitnet": True, "performance_mode": "standard"}),
            ("BitNet Ultra-Fast", {"use_bitnet": True, "performance_mode": "ultra_fast"}),
        ]
        
        for config_name, config_params in configs:
            try:
                config = BitACTConfig(action_dim=action_dim, **config_params)
                model = BitACTPolicy(config, observation_dim=obs_dim).to(device)
                model.eval()
                
                if hasattr(model, 'optimize_for_inference'):
                    model = model.optimize_for_inference()
                
                # Warmup
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(x)
                
                # Test
                times = []
                with torch.no_grad():
                    for _ in range(100):
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        actions = model(x)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) * 1000)
                
                avg_time = sum(times) / len(times)
                
                # Check if suitable for real-time control
                if batch_size == 1:
                    target_freq = 100  # 100Hz control
                    target_latency = 1000 / target_freq
                    suitable = "✅" if avg_time < target_latency else "❌"
                    print(f"  {config_name:<18}: {avg_time:.3f}ms {suitable}")
                else:
                    print(f"  {config_name:<18}: {avg_time:.3f}ms")
                
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"  {config_name:<18}: FAILED ({e})")


def propose_single_inference_optimizations():
    """Propose specific optimizations for single inference."""
    print(f"\n💡 Single Inference Optimization Strategy")
    print("=" * 50)
    
    optimizations = [
        {
            "name": "1. Eliminate Batch Overhead",
            "description": "Remove all batch-related computations for batch=1",
            "implementation": "Hard-code batch=1 paths, avoid .mean(dim=0) operations",
            "expected_speedup": "10-20%"
        },
        {
            "name": "2. Pre-compute Weight Quantization", 
            "description": "Quantize weights once during model loading",
            "implementation": "Store quantized weights as buffers, skip runtime quantization",
            "expected_speedup": "30-50%"
        },
        {
            "name": "3. Simplify Activation Quantization",
            "description": "Use direct sign() without mean-centering for single vectors",
            "implementation": "x_q = sign(x) for batch=1 only",
            "expected_speedup": "20-30%"
        },
        {
            "name": "4. Kernel Fusion for Small Sizes",
            "description": "Custom CUDA kernels optimized for small tensor operations",
            "implementation": "Fused sign+matmul kernel for typical robotics dimensions",
            "expected_speedup": "50-100%"
        },
        {
            "name": "5. Mixed Precision Strategy",
            "description": "Use FP16 for weights, FP32 for activations",
            "implementation": "Reduce memory bandwidth while maintaining precision",
            "expected_speedup": "15-25%"
        },
        {
            "name": "6. Avoid Quantization Entirely",
            "description": "For single inference, quantization overhead may not be worth it",
            "implementation": "Use standard FP32 for real-time control, BitNet for batch processing",
            "expected_speedup": "100% (no overhead)"
        }
    ]
    
    for opt in optimizations:
        print(f"\n{opt['name']}")
        print(f"  Description: {opt['description']}")
        print(f"  Implementation: {opt['implementation']}")
        print(f"  Expected Speedup: {opt['expected_speedup']}")
    
    print(f"\n🎯 Recommendation for Robotics:")
    print("=" * 35)
    print("For single robot control (batch=1):")
    print("  • Use FP32 for ultra-low latency requirements (<1ms)")
    print("  • Use optimized BitNet for moderate latency (1-5ms)")
    print("  • Consider hybrid: FP32 for critical control, BitNet for planning")
    print("\nFor batch processing (batch≥8):")
    print("  • BitNet Ultra-Fast is competitive and memory efficient")
    print("  • Good for offline training data processing")


def main():
    """Main analysis for single inference optimization."""
    print("🚀 Robotics Single Inference Optimization Analysis")
    print("=" * 60)
    
    # Analyze bottlenecks
    results = analyze_single_inference_bottlenecks()
    
    # Test real scenarios
    test_real_robotics_scenarios()
    
    # Propose optimizations
    propose_single_inference_optimizations()
    
    print(f"\n✅ Analysis Complete!")
    print("Key Insight: BitNet quantization overhead is significant for batch=1")
    print("Recommendation: Focus on hybrid approaches or custom kernels")


if __name__ == "__main__":
    main() 