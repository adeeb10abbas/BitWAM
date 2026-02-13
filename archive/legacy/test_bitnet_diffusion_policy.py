#!/usr/bin/env python3
"""
BitNet Diffusion Policy Integration Test

This script tests BitNet optimizations with LeRobot's diffusion policy,
focusing on the iterative denoising process where BitNet effects are amplified.
Diffusion policies are perfect for testing because they involve multiple forward
passes during inference, making performance differences more pronounced.
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
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Add local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our BitNet optimizations
from bit_vla.policies.bitact_policy import BitACTPolicy, BitACTConfig
from bit_vla.models.fast_bitlinear import FastBitLinear, UltraFastBitLinear


@dataclass
class DiffusionConfig:
    """Configuration for diffusion policy testing."""
    # Model architecture
    observation_dim: int = 32
    action_dim: int = 14
    diffusion_steps: int = 20  # Number of denoising steps
    noise_schedule: str = "cosine"
    
    # BitNet optimization levels
    optimization_levels: List[str] = None
    
    # Test configuration
    batch_sizes: List[int] = None
    num_inference_runs: int = 100
    warmup_runs: int = 10
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.optimization_levels is None:
            self.optimization_levels = ["standard", "fast", "ultra_fast"]
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8]


class SimplifiedBitDiffusionPolicy(nn.Module):
    """
    Simplified BitNet-optimized diffusion policy for robotics.
    
    This implements a streamlined diffusion model that focuses on the key
    components where BitNet optimizations matter most: the iterative denoising
    network that gets called multiple times during inference.
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
        
        # Create the denoising network with BitNet optimizations
        self.denoising_network = self._create_denoising_network(
            observation_dim, action_dim, hidden_dim, optimization_level
        )
        
        # Noise schedule
        self.register_buffer("betas", self._create_noise_schedule(diffusion_steps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        
        # For performance tracking
        self.total_forward_calls = 0
        self.total_inference_time = 0.0
    
    def _create_denoising_network(
        self, 
        obs_dim: int, 
        action_dim: int, 
        hidden_dim: int, 
        optimization_level: str
    ) -> nn.Module:
        """Create the core denoising network with BitNet optimizations."""
        
        # Input: noisy actions + observation + timestep embedding
        input_dim = action_dim + obs_dim + 64  # 64 for timestep embedding
        
        if optimization_level == "ultra_fast":
            # Maximum BitNet optimizations
            linear_class = UltraFastBitLinear
        elif optimization_level == "fast":
            # Balanced BitNet optimizations
            linear_class = FastBitLinear
        else:
            # Standard linear layers
            linear_class = nn.Linear
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Main denoising network - this gets called many times during inference!
        network = nn.Sequential(
            linear_class(input_dim, hidden_dim),
            nn.ReLU(),
            linear_class(hidden_dim, hidden_dim),
            nn.ReLU(),
            linear_class(hidden_dim, hidden_dim),
            nn.ReLU(),
            linear_class(hidden_dim, action_dim)  # Output: denoised action
        )
        
        return network
    
    def _create_noise_schedule(self, steps: int) -> torch.Tensor:
        """Create cosine noise schedule."""
        # Simple linear schedule for this test
        return torch.linspace(0.0001, 0.02, steps)
    
    def forward(self, observation: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Single forward pass of the denoising network.
        
        Args:
            observation: Environment observation [batch_size, obs_dim]
            timestep: Current diffusion timestep [batch_size, 1]
            
        Returns:
            Denoised action prediction [batch_size, action_dim]
        """
        batch_size = observation.shape[0]
        
        # For testing, we'll use a fixed timestep if not provided
        if timestep is None:
            timestep = torch.randint(0, self.diffusion_steps, (batch_size, 1), 
                                   device=observation.device, dtype=torch.float32)
        
        # Generate noisy action (in real diffusion this would be iteratively denoised)
        noisy_action = torch.randn(batch_size, self.action_dim, device=observation.device)
        
        # Time embedding
        time_emb = self.time_embed(timestep)
        
        # Concatenate inputs
        x = torch.cat([noisy_action, observation, time_emb], dim=-1)
        
        # Track forward calls
        self.total_forward_calls += 1
        
        # Denoising network forward pass
        denoised_action = self.denoising_network(x)
        
        return denoised_action
    
    def sample_action(self, observation: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """
        Full diffusion sampling process - this is where the magic happens!
        
        This method demonstrates the key advantage of testing diffusion policies:
        multiple forward passes amplify the BitNet performance differences.
        """
        if num_steps is None:
            num_steps = self.diffusion_steps
            
        batch_size = observation.shape[0]
        device = observation.device
        
        # Start with pure noise
        action = torch.randn(batch_size, self.action_dim, device=device)
        
        # Iterative denoising - THIS IS WHERE BITNET PERFORMANCE MATTERS!
        start_time = time.time()
        
        for t in reversed(range(num_steps)):
            # Create timestep tensor
            timestep = torch.full((batch_size, 1), t, device=device, dtype=torch.float32)
            
            # Denoise one step (this calls forward() which uses our BitNet layers)
            with torch.no_grad():
                denoised = self.forward(observation, timestep)
            
            # Update action (simplified DDPM update)
            if t > 0:
                noise = torch.randn_like(action)
                action = denoised + torch.sqrt(self.betas[t]) * noise
            else:
                action = denoised
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        
        return action


class DiffusionPolicyBenchmark:
    """Comprehensive benchmarking suite for BitNet diffusion policies."""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.results = {}
        
    def create_test_data(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create synthetic test data."""
        device = torch.device(self.config.device)
        
        # Create realistic robotics observation data
        observation = torch.randn(batch_size, self.config.observation_dim, device=device)
        
        # Create target actions
        target_action = torch.randn(batch_size, self.config.action_dim, device=device)
        
        return observation, target_action
    
    def benchmark_single_forward(self, model: SimplifiedBitDiffusionPolicy, batch_size: int) -> Dict[str, float]:
        """Benchmark single forward pass performance."""
        observation, _ = self.create_test_data(batch_size)
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = model(observation)
        
        # Actual benchmarking
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(self.config.num_inference_runs):
            with torch.no_grad():
                _ = model(observation)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / self.config.num_inference_runs
        throughput = self.config.num_inference_runs / total_time
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_fps": throughput,
            "total_time_s": total_time
        }
    
    def benchmark_full_diffusion(self, model: SimplifiedBitDiffusionPolicy, batch_size: int) -> Dict[str, float]:
        """Benchmark full diffusion sampling process."""
        observation, _ = self.create_test_data(batch_size)
        
        # Reset model counters
        model.total_forward_calls = 0
        model.total_inference_time = 0.0
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = model.sample_action(observation)
        
        # Reset counters after warmup
        model.total_forward_calls = 0
        model.total_inference_time = 0.0
        
        # Actual benchmarking
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(self.config.num_inference_runs):
            with torch.no_grad():
                _ = model.sample_action(observation)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / self.config.num_inference_runs
        throughput = self.config.num_inference_runs / total_time
        
        # Calculate per-step metrics
        total_forward_calls = model.total_forward_calls
        avg_forward_calls_per_sample = total_forward_calls / self.config.num_inference_runs
        avg_time_per_forward = model.total_inference_time / total_forward_calls if total_forward_calls > 0 else 0
        
        return {
            "avg_full_diffusion_time_ms": avg_time * 1000,
            "avg_forward_calls_per_sample": avg_forward_calls_per_sample,
            "avg_time_per_forward_ms": avg_time_per_forward * 1000,
            "total_forward_calls": total_forward_calls,
            "throughput_samples_per_sec": throughput
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark across all configurations."""
        results = {
            "config": {
                "observation_dim": self.config.observation_dim,
                "action_dim": self.config.action_dim,
                "diffusion_steps": self.config.diffusion_steps,
                "device": self.config.device,
                "num_inference_runs": self.config.num_inference_runs
            },
            "single_forward_results": {},
            "full_diffusion_results": {},
            "performance_analysis": {}
        }
        
        print("🚀 Starting Comprehensive BitNet Diffusion Policy Benchmark")
        print(f"📊 Testing {len(self.config.optimization_levels)} optimization levels")
        print(f"📏 Testing batch sizes: {self.config.batch_sizes}")
        print(f"🔄 {self.config.diffusion_steps} diffusion steps per sample")
        print(f"💻 Device: {self.config.device}")
        print()
        
        for opt_level in self.config.optimization_levels:
            print(f"🔧 Testing optimization level: {opt_level}")
            
            results["single_forward_results"][opt_level] = {}
            results["full_diffusion_results"][opt_level] = {}
            
            for batch_size in self.config.batch_sizes:
                print(f"  📦 Batch size: {batch_size}")
                
                # Create model with current optimization level
                model = SimplifiedBitDiffusionPolicy(
                    observation_dim=self.config.observation_dim,
                    action_dim=self.config.action_dim,
                    diffusion_steps=self.config.diffusion_steps,
                    optimization_level=opt_level
                ).to(self.config.device)
                
                # Benchmark single forward pass
                single_forward_results = self.benchmark_single_forward(model, batch_size)
                results["single_forward_results"][opt_level][batch_size] = single_forward_results
                
                # Benchmark full diffusion process
                full_diffusion_results = self.benchmark_full_diffusion(model, batch_size)
                results["full_diffusion_results"][opt_level][batch_size] = full_diffusion_results
                
                print(f"    Single Forward: {single_forward_results['avg_inference_time_ms']:.2f}ms")
                print(f"    Full Diffusion: {full_diffusion_results['avg_full_diffusion_time_ms']:.2f}ms")
                print(f"    Forward Calls/Sample: {full_diffusion_results['avg_forward_calls_per_sample']:.1f}")
        
        # Performance analysis
        results["performance_analysis"] = self._analyze_performance(results)
        
        return results
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance differences between optimization levels."""
        analysis = {
            "speedup_analysis": {},
            "efficiency_metrics": {},
            "batch_scaling": {}
        }
        
        # Compare against standard implementation
        standard_results = results["full_diffusion_results"].get("standard", {})
        
        for opt_level in self.config.optimization_levels:
            if opt_level == "standard":
                continue
                
            analysis["speedup_analysis"][opt_level] = {}
            
            for batch_size in self.config.batch_sizes:
                if batch_size not in standard_results:
                    continue
                    
                opt_results = results["full_diffusion_results"][opt_level][batch_size]
                std_results = standard_results[batch_size]
                
                speedup = std_results["avg_full_diffusion_time_ms"] / opt_results["avg_full_diffusion_time_ms"]
                
                analysis["speedup_analysis"][opt_level][batch_size] = {
                    "speedup": speedup,
                    "time_reduction_ms": std_results["avg_full_diffusion_time_ms"] - opt_results["avg_full_diffusion_time_ms"],
                    "throughput_improvement": opt_results["throughput_samples_per_sec"] / std_results["throughput_samples_per_sec"]
                }
        
        return analysis
    
    def save_results(self, results: Dict[str, Any], filename: str = "bitnet_diffusion_results.json"):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {filename}")
    
    def create_performance_plots(self, results: Dict[str, Any]):
        """Create performance visualization plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        batch_sizes = self.config.batch_sizes
        
        # Plot 1: Single Forward Pass Performance
        for opt_level in self.config.optimization_levels:
            times = [results["single_forward_results"][opt_level][bs]["avg_inference_time_ms"] 
                    for bs in batch_sizes]
            ax1.plot(batch_sizes, times, marker='o', label=f'{opt_level.title()}')
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Average Time (ms)')
        ax1.set_title('Single Forward Pass Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Full Diffusion Performance
        for opt_level in self.config.optimization_levels:
            times = [results["full_diffusion_results"][opt_level][bs]["avg_full_diffusion_time_ms"] 
                    for bs in batch_sizes]
            ax2.plot(batch_sizes, times, marker='s', label=f'{opt_level.title()}')
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Average Time (ms)')
        ax2.set_title('Full Diffusion Process Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Throughput Comparison
        for opt_level in self.config.optimization_levels:
            throughputs = [results["full_diffusion_results"][opt_level][bs]["throughput_samples_per_sec"] 
                          for bs in batch_sizes]
            ax3.plot(batch_sizes, throughputs, marker='^', label=f'{opt_level.title()}')
        
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Throughput (samples/sec)')
        ax3.set_title('Diffusion Sampling Throughput')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Speedup Analysis (Batch=1 focus)
        if "performance_analysis" in results and "speedup_analysis" in results["performance_analysis"]:
            opt_levels = [level for level in self.config.optimization_levels if level != "standard"]
            batch_1_speedups = []
            
            for opt_level in opt_levels:
                if 1 in results["performance_analysis"]["speedup_analysis"].get(opt_level, {}):
                    speedup = results["performance_analysis"]["speedup_analysis"][opt_level][1]["speedup"]
                    batch_1_speedups.append(speedup)
                else:
                    batch_1_speedups.append(1.0)
            
            bars = ax4.bar(opt_levels, batch_1_speedups, color=['orange', 'red'][:len(opt_levels)])
            ax4.set_ylabel('Speedup Factor')
            ax4.set_title('Speedup vs Standard (Batch=1)')
            ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, speedup in zip(bars, batch_1_speedups):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{speedup:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('bitnet_diffusion_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Performance plots saved as 'bitnet_diffusion_performance.png'")


def main():
    """Main execution function."""
    print("🤖 BitNet Diffusion Policy Integration Test")
    print("=" * 60)
    
    # Configuration
    config = DiffusionConfig(
        observation_dim=32,
        action_dim=14,
        diffusion_steps=20,  # Typical for robotics diffusion policies
        batch_sizes=[1, 2, 4, 8],  # Focus on small batches for robotics
        num_inference_runs=50,  # Reduced for faster testing
        warmup_runs=5
    )
    
    # Run benchmark
    benchmark = DiffusionPolicyBenchmark(config)
    results = benchmark.run_comprehensive_benchmark()
    
    # Save and visualize results
    benchmark.save_results(results)
    benchmark.create_performance_plots(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 SUMMARY: BitNet Diffusion Policy Performance")
    print("=" * 60)
    
    if "performance_analysis" in results and "speedup_analysis" in results["performance_analysis"]:
        for opt_level in ["fast", "ultra_fast"]:
            if opt_level in results["performance_analysis"]["speedup_analysis"]:
                batch_1_analysis = results["performance_analysis"]["speedup_analysis"][opt_level].get(1, {})
                if batch_1_analysis:
                    speedup = batch_1_analysis["speedup"]
                    time_saved = batch_1_analysis["time_reduction_ms"]
                    
                    print(f"\n🚀 {opt_level.upper()} Optimization (Batch=1):")
                    print(f"   Speedup: {speedup:.2f}x")
                    print(f"   Time Saved: {time_saved:.2f}ms per diffusion sample")
                    print(f"   Time Saved per Forward: {time_saved/config.diffusion_steps:.2f}ms")
    
    # Key insight about diffusion policies
    print(f"\n💡 KEY INSIGHT:")
    print(f"   Diffusion policies amplify BitNet effects through {config.diffusion_steps} forward passes")
    print(f"   Each diffusion sample requires {config.diffusion_steps} denoising steps")
    print(f"   Small per-step improvements become significant overall gains!")
    
    print(f"\n📊 Real-world Impact:")
    batch_1_std = results["full_diffusion_results"]["standard"][1]["avg_full_diffusion_time_ms"]
    if "ultra_fast" in results["full_diffusion_results"]:
        batch_1_ultra = results["full_diffusion_results"]["ultra_fast"][1]["avg_full_diffusion_time_ms"]
        robot_freq = 1000 / batch_1_ultra  # Hz
        print(f"   Standard: {batch_1_std:.1f}ms per action")
        print(f"   Ultra-Fast: {batch_1_ultra:.1f}ms per action") 
        print(f"   → Enables {robot_freq:.0f}Hz robot control frequency")


if __name__ == "__main__":
    main() 