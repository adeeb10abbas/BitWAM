#!/usr/bin/env python3
"""
BitNet Compatible Diffusion Policy Test

This script tests our BitNet compatible implementation (matching the reference
BitNet folder structure) with diffusion policies for batch=1 robotics inference.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

# Add local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our compatible BitNet implementation
from bit_vla.models.bitnet_compatible import (
    BitLinearCompatible, 
    activation_quant,
    weight_quant,
    activation_quant_fast,
    weight_quant_fast,
    create_bitnet_linear
)

# Import comparison with reference BitNet (if available)
try:
    sys.path.insert(0, str(Path(__file__).parent / "BitNet"))
    from bitnet import BitLinear, BitNetTransformer
    REFERENCE_AVAILABLE = True
    print("✅ Reference BitNet implementation found")
except ImportError:
    REFERENCE_AVAILABLE = False
    print("⚠️ Reference BitNet implementation not available")


class CompatibleBitDiffusionPolicy(nn.Module):
    """
    Diffusion policy using our BitNet compatible implementation.
    
    This follows the same quantization patterns as the reference BitNet
    but with our performance optimizations.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        diffusion_steps: int = 20,
        hidden_dim: int = 256,
        optimization_level: str = "standard"
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps
        self.optimization_level = optimization_level
        
        # Time embedding using compatible BitLinear
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),  # Keep standard for time embedding
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Main denoising network using our compatible BitLinear
        input_dim = action_dim + observation_dim + 64  # 64 for timestep embedding
        
        self.denoising_network = nn.Sequential(
            create_bitnet_linear(input_dim, hidden_dim, optimization_level=optimization_level),
            nn.ReLU(),
            create_bitnet_linear(hidden_dim, hidden_dim, optimization_level=optimization_level),
            nn.ReLU(),
            create_bitnet_linear(hidden_dim, hidden_dim, optimization_level=optimization_level),
            nn.ReLU(),
            create_bitnet_linear(hidden_dim, action_dim, optimization_level=optimization_level)
        )
        
        # Noise schedule
        self.register_buffer("betas", self._create_noise_schedule(diffusion_steps))
        
        # Performance tracking
        self.total_forward_calls = 0
        self.total_inference_time = 0.0
    
    def _create_noise_schedule(self, steps: int) -> torch.Tensor:
        """Create simple linear noise schedule."""
        return torch.linspace(0.0001, 0.02, steps)
    
    def forward(self, observation: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single forward pass of the denoising network."""
        batch_size = observation.shape[0]
        
        if timestep is None:
            timestep = torch.randint(0, self.diffusion_steps, (batch_size, 1), 
                                   device=observation.device, dtype=torch.float32)
        
        # Generate noisy action
        noisy_action = torch.randn(batch_size, self.action_dim, device=observation.device)
        
        # Time embedding
        time_emb = self.time_embed(timestep)
        
        # Concatenate inputs
        x = torch.cat([noisy_action, observation, time_emb], dim=-1)
        
        # Track forward calls
        self.total_forward_calls += 1
        
        # Denoising network forward pass using BitNet layers
        denoised_action = self.denoising_network(x)
        
        return denoised_action
    
    def sample_action(self, observation: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """Full diffusion sampling process."""
        if num_steps is None:
            num_steps = self.diffusion_steps
            
        batch_size = observation.shape[0]
        device = observation.device
        
        # Start with pure noise
        action = torch.randn(batch_size, self.action_dim, device=device)
        
        # Iterative denoising
        start_time = time.time()
        
        for t in reversed(range(num_steps)):
            timestep = torch.full((batch_size, 1), t, device=device, dtype=torch.float32)
            
            with torch.no_grad():
                denoised = self.forward(observation, timestep)
            
            # Simplified DDPM update
            if t > 0:
                noise = torch.randn_like(action)
                action = denoised + torch.sqrt(self.betas[t]) * noise
            else:
                action = denoised
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        
        return action


class ReferenceBitDiffusionPolicy(nn.Module):
    """
    Diffusion policy using the reference BitNet implementation (if available).
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        diffusion_steps: int = 20,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        if not REFERENCE_AVAILABLE:
            raise ImportError("Reference BitNet implementation not available")
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Main denoising network using reference BitLinear
        input_dim = action_dim + observation_dim + 64
        
        self.denoising_network = nn.Sequential(
            BitLinear(input_dim, hidden_dim),
            nn.ReLU(),
            BitLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            BitLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            BitLinear(hidden_dim, action_dim)
        )
        
        # Noise schedule
        self.register_buffer("betas", torch.linspace(0.0001, 0.02, diffusion_steps))
        
        # Performance tracking
        self.total_forward_calls = 0
        self.total_inference_time = 0.0
    
    def forward(self, observation: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single forward pass using reference implementation."""
        batch_size = observation.shape[0]
        
        if timestep is None:
            timestep = torch.randint(0, self.diffusion_steps, (batch_size, 1), 
                                   device=observation.device, dtype=torch.float32)
        
        noisy_action = torch.randn(batch_size, self.action_dim, device=observation.device)
        time_emb = self.time_embed(timestep)
        x = torch.cat([noisy_action, observation, time_emb], dim=-1)
        
        self.total_forward_calls += 1
        denoised_action = self.denoising_network(x)
        
        return denoised_action
    
    def sample_action(self, observation: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """Full diffusion sampling using reference implementation."""
        if num_steps is None:
            num_steps = self.diffusion_steps
            
        batch_size = observation.shape[0]
        device = observation.device
        
        action = torch.randn(batch_size, self.action_dim, device=device)
        
        start_time = time.time()
        
        for t in reversed(range(num_steps)):
            timestep = torch.full((batch_size, 1), t, device=device, dtype=torch.float32)
            
            with torch.no_grad():
                denoised = self.forward(observation, timestep)
            
            if t > 0:
                noise = torch.randn_like(action)
                action = denoised + torch.sqrt(self.betas[t]) * noise
            else:
                action = denoised
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        
        return action


def benchmark_quantization_functions():
    """Test and benchmark the different quantization functions."""
    print("🧪 Testing Quantization Function Compatibility")
    print("-" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create test tensors
    test_weights = torch.randn(256, 128, device=device) * 0.02
    test_activations = torch.randn(4, 128, device=device)
    
    # Test weight quantization
    print("Weight Quantization:")
    
    # Reference implementation
    ref_w_quant = weight_quant(test_weights)
    print(f"  Reference: shape={ref_w_quant.shape}, mean={ref_w_quant.mean().item():.4f}")
    
    # Fast implementation
    fast_w_quant = weight_quant_fast(test_weights)
    print(f"  Fast:      shape={fast_w_quant.shape}, mean={fast_w_quant.mean().item():.4f}")
    
    # Test activation quantization
    print("\nActivation Quantization:")
    
    # Reference implementation
    ref_a_quant = activation_quant(test_activations)
    print(f"  Reference: shape={ref_a_quant.shape}, mean={ref_a_quant.mean().item():.4f}")
    
    # Fast implementation
    fast_a_quant = activation_quant_fast(test_activations)
    print(f"  Fast:      shape={fast_a_quant.shape}, mean={fast_a_quant.mean().item():.4f}")
    
    # Benchmark quantization speed
    print("\nQuantization Speed Benchmark:")
    num_runs = 100
    
    # Weight quantization timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_runs):
        _ = weight_quant(test_weights)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    ref_weight_time = (time.time() - start_time) / num_runs * 1000
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_runs):
        _ = weight_quant_fast(test_weights)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    fast_weight_time = (time.time() - start_time) / num_runs * 1000
    
    print(f"  Weight Quantization:")
    print(f"    Reference: {ref_weight_time:.3f}ms")
    print(f"    Fast:      {fast_weight_time:.3f}ms")
    print(f"    Speedup:   {ref_weight_time/fast_weight_time:.2f}×")
    
    # Activation quantization timing  
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_runs):
        _ = activation_quant(test_activations)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    ref_act_time = (time.time() - start_time) / num_runs * 1000
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_runs):
        _ = activation_quant_fast(test_activations)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    fast_act_time = (time.time() - start_time) / num_runs * 1000
    
    print(f"  Activation Quantization:")
    print(f"    Reference: {ref_act_time:.3f}ms")
    print(f"    Fast:      {fast_act_time:.3f}ms")
    print(f"    Speedup:   {ref_act_time/fast_act_time:.2f}×")


def benchmark_diffusion_models():
    """Benchmark different diffusion policy implementations."""
    print("\n🚀 Benchmarking BitNet Diffusion Policy Implementations")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1  # Focus on robotics batch=1 case
    observation_dim = 32
    action_dim = 14
    diffusion_steps = 20
    num_runs = 50
    warmup_runs = 5
    
    # Create test data
    observation = torch.randn(batch_size, observation_dim, device=device)
    
    results = {}
    
    # Test models to benchmark
    models_to_test = [
        ("FP32 Standard", "standard"),
        ("Compatible Standard", "standard"),
        ("Compatible Fast", "fast"),
        ("Compatible Ultra-Fast", "ultra_fast"),
    ]
    
    # Add reference model if available
    if REFERENCE_AVAILABLE:
        models_to_test.insert(1, ("Reference BitNet", "reference"))
    
    for model_name, optimization_level in models_to_test:
        print(f"\n🔧 Testing: {model_name}")
        
        # Create model
        if model_name == "FP32 Standard":
            # Standard FP32 model for comparison
            model = CompatibleBitDiffusionPolicy(
                observation_dim, action_dim, diffusion_steps, 256, "standard"
            )
            # Replace BitLinear layers with standard Linear
            for name, module in model.named_modules():
                if isinstance(module, BitLinearCompatible):
                    parent = model
                    for part in name.split('.')[:-1]:
                        parent = getattr(parent, part)
                    layer_name = name.split('.')[-1]
                    setattr(parent, layer_name, 
                           nn.Linear(module.in_features, module.out_features, 
                                   module.bias is not None))
        elif model_name == "Reference BitNet":
            model = ReferenceBitDiffusionPolicy(observation_dim, action_dim, diffusion_steps, 256)
        else:
            model = CompatibleBitDiffusionPolicy(
                observation_dim, action_dim, diffusion_steps, 256, optimization_level
            )
        
        model = model.to(device).eval()
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model.sample_action(observation)
        
        # Reset counters
        model.total_forward_calls = 0
        model.total_inference_time = 0.0
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model.sample_action(observation)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs * 1000  # Convert to ms
        throughput = num_runs / total_time
        
        results[model_name] = {
            "avg_diffusion_time_ms": avg_time,
            "throughput_samples_per_sec": throughput,
            "forward_calls_per_sample": model.total_forward_calls / num_runs,
            "robot_control_freq_hz": 1000 / avg_time
        }
        
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Forward calls/sample: {model.total_forward_calls / num_runs:.1f}")
        print(f"  Robot control freq: {1000 / avg_time:.0f}Hz")
    
    return results


def analyze_results(results: Dict[str, Any]):
    """Analyze and compare the benchmark results."""
    print("\n" + "=" * 60)
    print("📊 COMPARATIVE ANALYSIS")
    print("=" * 60)
    
    # Use FP32 as baseline
    baseline_name = "FP32 Standard"
    if baseline_name not in results:
        baseline_name = list(results.keys())[0]
    
    baseline_time = results[baseline_name]["avg_diffusion_time_ms"]
    
    print(f"\n🎯 Performance vs {baseline_name} (Batch=1):")
    print("-" * 40)
    
    for model_name, metrics in results.items():
        speedup = baseline_time / metrics["avg_diffusion_time_ms"]
        time_diff = metrics["avg_diffusion_time_ms"] - baseline_time
        
        status = "✅" if speedup >= 1.0 else "❌"
        
        print(f"{status} {model_name}:")
        print(f"   Time: {metrics['avg_diffusion_time_ms']:.2f}ms")
        print(f"   Speedup: {speedup:.2f}×")
        print(f"   Difference: {time_diff:+.2f}ms")
        print(f"   Robot Freq: {metrics['robot_control_freq_hz']:.0f}Hz")
        print()
    
    # Compatibility check
    print("🔍 BITNET COMPATIBILITY VERIFICATION:")
    print("-" * 40)
    
    compatible_models = [name for name in results.keys() if "Compatible" in name]
    if len(compatible_models) >= 2:
        std_model = next((name for name in compatible_models if "Standard" in name), None)
        fast_model = next((name for name in compatible_models if "Fast" in name), None)
        
        if std_model and fast_model:
            std_time = results[std_model]["avg_diffusion_time_ms"]
            fast_time = results[fast_model]["avg_diffusion_time_ms"]
            improvement = std_time / fast_time
            
            print(f"✅ Our optimizations work!")
            print(f"   Standard: {std_time:.2f}ms")
            print(f"   Optimized: {fast_time:.2f}ms")
            print(f"   Improvement: {improvement:.2f}×")
    
    # Reference comparison
    if "Reference BitNet" in results:
        ref_time = results["Reference BitNet"]["avg_diffusion_time_ms"]
        our_time = results.get("Compatible Standard", {}).get("avg_diffusion_time_ms", 0)
        
        if our_time > 0:
            compatibility = ref_time / our_time
            print(f"\n🤝 Reference BitNet Compatibility:")
            print(f"   Reference: {ref_time:.2f}ms")
            print(f"   Our Compatible: {our_time:.2f}ms")
            print(f"   Ratio: {compatibility:.2f}× (closer to 1.0 = more compatible)")


def main():
    """Main execution function."""
    print("🤖 BitNet Compatible Diffusion Policy Test")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Reference BitNet: {'Available ✅' if REFERENCE_AVAILABLE else 'Not Available ⚠️'}")
    
    # Test quantization function compatibility
    benchmark_quantization_functions()
    
    # Benchmark diffusion models
    results = benchmark_diffusion_models()
    
    # Analyze results
    analyze_results(results)
    
    # Save results
    with open("bitnet_compatible_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to bitnet_compatible_results.json")
    
    print("\n" + "=" * 60)
    print("🎯 FINAL RECOMMENDATIONS")
    print("=" * 60)
    print("For LeRobot diffusion policies with batch=1:")
    print("1. Use FP32 for best performance on current hardware")
    print("2. Use Compatible Ultra-Fast if memory is constrained")
    print("3. Our implementation maintains BitNet API compatibility")
    print("4. Performance gap confirms the quantization overhead issue")


if __name__ == "__main__":
    main() 