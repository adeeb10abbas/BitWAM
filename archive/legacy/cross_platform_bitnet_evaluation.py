#!/usr/bin/env python3
"""
Cross-Platform BitNet Diffusion Policy Evaluation

This script evaluates BitNet performance across different hardware platforms:
- NVIDIA GPUs (CUDA)
- Apple ARM M chips (MPS/CPU)
- Intel/AMD CPUs

Automatically detects platform, runs benchmarks, saves results, and enables
cross-platform performance comparison.
"""

import os
import sys
import time
import json
import platform
import subprocess
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import psutil

# Add local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our BitNet implementations
try:
    from bit_vla.models.bitnet_compatible import (
        BitLinearCompatible, 
        activation_quant,
        weight_quant,
        activation_quant_fast,
        weight_quant_fast,
        create_bitnet_linear
    )
    BITNET_AVAILABLE = True
except ImportError:
    BITNET_AVAILABLE = False
    print("⚠️ BitNet compatible implementation not found")


class PlatformDetector:
    """Detect and characterize the current hardware platform."""
    
    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """Get comprehensive platform information."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }
        
        # Basic system info
        info.update({
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        })
        
        # Detect hardware type
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            info["hardware_type"] = "apple_arm"
            info["chip_info"] = PlatformDetector._get_apple_chip_info()
        elif torch.cuda.is_available():
            info["hardware_type"] = "nvidia_gpu"
            info["gpu_info"] = PlatformDetector._get_nvidia_gpu_info()
        else:
            info["hardware_type"] = "cpu_only"
            info["cpu_info"] = PlatformDetector._get_cpu_info()
        
        # Detect compute device
        info["compute_device"] = PlatformDetector._get_compute_device()
        
        return info
    
    @staticmethod
    def _get_apple_chip_info() -> Dict[str, Any]:
        """Get Apple Silicon chip information."""
        try:
            # Try to get chip info from system_profiler
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"], 
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.split('\n')
            chip_info = {}
            
            for line in lines:
                if "Chip:" in line:
                    chip_info["chip_name"] = line.split(":")[-1].strip()
                elif "Total Number of Cores:" in line:
                    chip_info["total_cores"] = line.split(":")[-1].strip()
                elif "Memory:" in line:
                    chip_info["unified_memory"] = line.split(":")[-1].strip()
            
            return chip_info
        except Exception:
            return {"chip_name": "Apple Silicon (unknown)", "detection_method": "fallback"}
    
    @staticmethod
    def _get_nvidia_gpu_info() -> Dict[str, Any]:
        """Get NVIDIA GPU information."""
        if not torch.cuda.is_available():
            return {}
        
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "compute_capability": torch.cuda.get_device_properties(0).major,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        
        return gpu_info
    
    @staticmethod
    def _get_cpu_info() -> Dict[str, Any]:
        """Get CPU information for non-GPU systems."""
        return {
            "cpu_freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else "unknown",
            "cpu_freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else "unknown",
        }
    
    @staticmethod
    def _get_compute_device() -> str:
        """Determine the best compute device for this platform."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


class CrossPlatformBitDiffusionPolicy(nn.Module):
    """BitNet diffusion policy optimized for cross-platform evaluation."""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        diffusion_steps: int = 20,
        hidden_dim: int = 256,
        optimization_level: str = "standard",
        device: str = "auto"
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps
        self.optimization_level = optimization_level
        
        # Auto-detect device if not specified
        if device == "auto":
            device = PlatformDetector._get_compute_device()
        self.device = device
        
        # Time embedding (keep as FP32 for stability across platforms)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Main denoising network
        input_dim = action_dim + observation_dim + 64
        
        if BITNET_AVAILABLE:
            self.denoising_network = nn.Sequential(
                create_bitnet_linear(input_dim, hidden_dim, optimization_level=optimization_level),
                nn.ReLU(),
                create_bitnet_linear(hidden_dim, hidden_dim, optimization_level=optimization_level),
                nn.ReLU(),
                create_bitnet_linear(hidden_dim, hidden_dim, optimization_level=optimization_level),
                nn.ReLU(),
                create_bitnet_linear(hidden_dim, action_dim, optimization_level=optimization_level)
            )
        else:
            # Fallback to standard linear layers if BitNet not available
            self.denoising_network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        # Noise schedule
        self.register_buffer("betas", torch.linspace(0.0001, 0.02, diffusion_steps))
        
        # Performance tracking
        self.total_forward_calls = 0
        self.total_inference_time = 0.0
        
    def forward(self, observation: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the denoising network."""
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
        
        # Denoising network forward pass
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


class CrossPlatformBenchmark:
    """Benchmark runner for cross-platform evaluation."""
    
    def __init__(self, save_dir: str = "cross_platform_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.platform_info = PlatformDetector.get_platform_info()
        
    def run_comprehensive_benchmark(
        self,
        observation_dim: int = 32,
        action_dim: int = 14,
        diffusion_steps: int = 20,
        hidden_dim: int = 256,
        num_runs: int = 50,
        warmup_runs: int = 5
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across all available configurations."""
        
        print(f"🚀 Cross-Platform BitNet Evaluation")
        print(f"Platform: {self.platform_info['hardware_type']}")
        print(f"Device: {self.platform_info['compute_device']}")
        print(f"System: {self.platform_info['system']} {self.platform_info['machine']}")
        
        device = self.platform_info['compute_device']
        results = {
            "platform_info": self.platform_info,
            "benchmark_config": {
                "observation_dim": observation_dim,
                "action_dim": action_dim,
                "diffusion_steps": diffusion_steps,
                "hidden_dim": hidden_dim,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            },
            "model_results": {}
        }
        
        # Test configurations
        if BITNET_AVAILABLE:
            test_configs = [
                ("FP32_Standard", "fp32", None),
                ("BitNet_Standard", "bitnet", "standard"),
                ("BitNet_Fast", "bitnet", "fast"),
                ("BitNet_Ultra_Fast", "bitnet", "ultra_fast"),
            ]
        else:
            test_configs = [("FP32_Standard", "fp32", None)]
        
        # Create test observation
        observation = torch.randn(1, observation_dim, device=device)
        
        for config_name, model_type, optimization_level in test_configs:
            print(f"\n🔧 Testing: {config_name}")
            
            try:
                # Create model
                if model_type == "fp32":
                    model = self._create_fp32_model(observation_dim, action_dim, diffusion_steps, hidden_dim)
                else:
                    model = CrossPlatformBitDiffusionPolicy(
                        observation_dim, action_dim, diffusion_steps, hidden_dim, 
                        optimization_level, device
                    )
                
                model = model.to(device).eval()
                
                # Warmup
                print(f"   Warming up...")
                for _ in range(warmup_runs):
                    with torch.no_grad():
                        _ = model.sample_action(observation)
                
                # Reset counters
                model.total_forward_calls = 0
                model.total_inference_time = 0.0
                
                # Benchmark
                print(f"   Running {num_runs} iterations...")
                
                # Synchronize based on device
                self._device_sync(device)
                start_time = time.time()
                
                for _ in range(num_runs):
                    with torch.no_grad():
                        _ = model.sample_action(observation)
                
                self._device_sync(device)
                end_time = time.time()
                
                total_time = end_time - start_time
                avg_time = total_time / num_runs * 1000  # Convert to ms
                throughput = num_runs / total_time
                
                results["model_results"][config_name] = {
                    "avg_diffusion_time_ms": avg_time,
                    "throughput_samples_per_sec": throughput,
                    "forward_calls_per_sample": model.total_forward_calls / num_runs,
                    "robot_control_freq_hz": 1000 / avg_time,
                    "total_benchmark_time_sec": total_time,
                }
                
                print(f"   ✅ Average time: {avg_time:.2f}ms")
                print(f"   ✅ Throughput: {throughput:.1f} samples/sec")
                print(f"   ✅ Robot freq: {1000 / avg_time:.0f}Hz")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results["model_results"][config_name] = {"error": str(e)}
        
        return results
    
    def _create_fp32_model(self, obs_dim, act_dim, diff_steps, hidden_dim):
        """Create FP32 baseline model."""
        model = CrossPlatformBitDiffusionPolicy(
            obs_dim, act_dim, diff_steps, hidden_dim, "standard"
        )
        
        # Replace BitLinear layers with standard Linear layers
        def replace_bitlinear(module):
            for name, child in module.named_children():
                if isinstance(child, BitLinearCompatible):
                    setattr(module, name, nn.Linear(
                        child.in_features, child.out_features, child.bias is not None
                    ))
                else:
                    replace_bitlinear(child)
        
        if BITNET_AVAILABLE:
            replace_bitlinear(model)
        
        return model
    
    def _device_sync(self, device: str):
        """Synchronize based on device type."""
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
        # CPU doesn't need synchronization
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save results to file with platform-specific naming."""
        platform_type = results["platform_info"]["hardware_type"]
        device_type = results["platform_info"]["compute_device"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"bitnet_eval_{platform_type}_{device_type}_{timestamp}.json"
        filepath = self.save_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
        return str(filepath)
    
    def compare_platforms(self, result_files: List[str]) -> Dict[str, Any]:
        """Compare results across multiple platforms."""
        if len(result_files) < 2:
            print("⚠️ Need at least 2 result files for comparison")
            return {}
        
        print(f"\n📊 Cross-Platform Comparison")
        print("=" * 50)
        
        # Load all results
        all_results = []
        for file_path in result_files:
            with open(file_path, 'r') as f:
                all_results.append(json.load(f))
        
        # Create comparison
        comparison = {
            "comparison_timestamp": datetime.now().isoformat(),
            "platforms": [],
            "model_comparison": {}
        }
        
        # Extract platform info
        for result in all_results:
            platform_info = {
                "hardware_type": result["platform_info"]["hardware_type"],
                "compute_device": result["platform_info"]["compute_device"],
                "system": result["platform_info"]["system"],
                "machine": result["platform_info"]["machine"],
            }
            
            # Add chip-specific info
            if "chip_info" in result["platform_info"]:
                platform_info["chip_info"] = result["platform_info"]["chip_info"]
            elif "gpu_info" in result["platform_info"]:
                platform_info["gpu_info"] = result["platform_info"]["gpu_info"]
            
            comparison["platforms"].append(platform_info)
        
        # Compare model performance
        # Get all model names from first result
        model_names = list(all_results[0]["model_results"].keys())
        
        for model_name in model_names:
            comparison["model_comparison"][model_name] = []
            
            for i, result in enumerate(all_results):
                if model_name in result["model_results"]:
                    model_result = result["model_results"][model_name].copy()
                    model_result["platform_index"] = i
                    comparison["model_comparison"][model_name].append(model_result)
        
        # Print comparison summary
        self._print_comparison_summary(comparison)
        
        # Save comparison
        comparison_file = self.save_dir / f"platform_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n💾 Comparison saved to: {comparison_file}")
        return comparison
    
    def _print_comparison_summary(self, comparison: Dict[str, Any]):
        """Print a summary of the platform comparison."""
        platforms = comparison["platforms"]
        model_comparison = comparison["model_comparison"]
        
        print(f"\n🏆 Performance Summary")
        print("-" * 40)
        
        for model_name, results in model_comparison.items():
            if not results or any("error" in r for r in results):
                continue
                
            print(f"\n{model_name}:")
            
            for i, result in enumerate(results):
                platform = platforms[i]
                time_ms = result.get("avg_diffusion_time_ms", 0)
                freq_hz = result.get("robot_control_freq_hz", 0)
                
                print(f"  {platform['hardware_type']} ({platform['compute_device']}): "
                      f"{time_ms:.2f}ms, {freq_hz:.0f}Hz")
            
            # Find best performer
            best_idx = min(range(len(results)), 
                          key=lambda i: results[i].get("avg_diffusion_time_ms", float('inf')))
            best_platform = platforms[best_idx]
            best_time = results[best_idx]["avg_diffusion_time_ms"]
            
            print(f"  🥇 Fastest: {best_platform['hardware_type']} at {best_time:.2f}ms")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Platform BitNet Evaluation")
    parser.add_argument("--mode", choices=["benchmark", "compare"], default="benchmark",
                       help="Run benchmark or compare existing results")
    parser.add_argument("--compare-files", nargs="+", 
                       help="Result files to compare (for compare mode)")
    parser.add_argument("--save-dir", default="cross_platform_results",
                       help="Directory to save results")
    parser.add_argument("--runs", type=int, default=50,
                       help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=5,
                       help="Number of warmup runs")
    
    args = parser.parse_args()
    
    benchmark = CrossPlatformBenchmark(args.save_dir)
    
    if args.mode == "benchmark":
        # Run benchmark
        results = benchmark.run_comprehensive_benchmark(
            num_runs=args.runs, warmup_runs=args.warmup
        )
        result_file = benchmark.save_results(results)
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Copy this script to another machine (Apple Silicon or different GPU)")
        print(f"2. Run: python cross_platform_bitnet_evaluation.py --mode benchmark")
        print(f"3. Compare results: python cross_platform_bitnet_evaluation.py --mode compare --compare-files {result_file} <other_result_file>")
        
    elif args.mode == "compare":
        if not args.compare_files or len(args.compare_files) < 2:
            print("❌ Need at least 2 result files for comparison")
            print("Usage: --mode compare --compare-files file1.json file2.json")
            sys.exit(1)
        
        # Compare platforms
        comparison = benchmark.compare_platforms(args.compare_files)
        
        if comparison:
            print(f"\n✅ Cross-platform comparison complete!")


if __name__ == "__main__":
    main() 