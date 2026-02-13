#!/usr/bin/env python3
"""
Comprehensive Speed Benchmarking for 1bit_vla Models

This script focuses on inference speed comparison between BitNet and standard VLA models.
Key metrics:
- Single inference latency
- Batch inference throughput  
- Memory bandwidth utilization
- Real-world robotics scenarios
- GPU vs CPU performance

Usage:
    python examples/speed_benchmark.py --model bitact --batch_sizes 1,8,32 --device cuda
    python examples/speed_benchmark.py --comprehensive --save_results
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import time
import json
import argparse
import numpy as np
from contextlib import contextmanager
import gc

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bit_vla import BitACTPolicy, BitACTConfig, print_model_info


@contextmanager
def timer():
    """Context manager for precise timing measurements"""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    

def warmup_model(model, input_shape, device, num_warmup=10):
    """Warm up model for consistent timing measurements"""
    model.eval()
    dummy_input = torch.randn(input_shape, device=device)
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Clear cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def measure_single_inference(model, input_shape, device, num_runs=100):
    """Measure single inference latency with statistical analysis"""
    model.eval()
    warmup_model(model, input_shape, device)
    
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            dummy_input = torch.randn(input_shape, device=device)
            
            # Synchronize for accurate GPU timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            with timer() as get_time:
                output = model(dummy_input)
                
                # Synchronize again
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            times.append(get_time() * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'raw_times': times
    }


def measure_batch_throughput(model, obs_dim, device, batch_sizes=[1, 8, 16, 32, 64], num_runs=50):
    """Measure batch inference throughput for different batch sizes"""
    model.eval()
    results = {}
    
    for batch_size in batch_sizes:
        input_shape = (batch_size, obs_dim)
        warmup_model(model, input_shape, device, num_warmup=5)
        
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                dummy_input = torch.randn(input_shape, device=device)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                output = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        throughput = batch_size / avg_time  # samples per second
        
        results[batch_size] = {
            'avg_time_ms': avg_time * 1000,
            'throughput_sps': throughput,
            'time_per_sample_ms': (avg_time * 1000) / batch_size
        }
    
    return results


def measure_memory_bandwidth(model, obs_dim, device, duration_seconds=2.0):
    """Measure memory bandwidth utilization during inference"""
    model.eval()
    batch_size = 32
    input_shape = (batch_size, obs_dim)
    
    warmup_model(model, input_shape, device)
    
    if device.type == 'cuda':
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
    
    num_inferences = 0
    
    with torch.no_grad():
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < duration_seconds:
            dummy_input = torch.randn(input_shape, device=device)
            output = model(dummy_input)
            num_inferences += 1
    
    total_time = time.perf_counter() - start_time
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = peak_memory - initial_memory
        
        return {
            'total_inferences': num_inferences,
            'inferences_per_second': num_inferences / total_time,
            'memory_used_mb': memory_used / (1024 * 1024),
            'peak_memory_mb': peak_memory / (1024 * 1024),
            'memory_per_inference_mb': memory_used / (1024 * 1024) / num_inferences
        }
    else:
        return {
            'total_inferences': num_inferences,
            'inferences_per_second': num_inferences / total_time,
            'memory_used_mb': None,
            'peak_memory_mb': None,
            'memory_per_inference_mb': None
        }


def robotics_scenario_benchmark(model, obs_dim, device):
    """Benchmark real-world robotics scenarios"""
    model.eval()
    scenarios = {
        'real_time_control': {
            'batch_size': 1,
            'target_hz': 10,  # 10 Hz control loop
            'description': 'Single robot real-time control'
        },
        'multi_robot': {
            'batch_size': 4,
            'target_hz': 10,
            'description': '4 robots simultaneous control'
        },
        'high_frequency': {
            'batch_size': 1,
            'target_hz': 50,  # High frequency control
            'description': 'High-frequency single robot control'
        },
        'batch_processing': {
            'batch_size': 32,
            'target_hz': 1,   # Batch processing scenario
            'description': 'Offline batch trajectory processing'
        }
    }
    
    results = {}
    
    for scenario_name, config in scenarios.items():
        batch_size = config['batch_size']
        target_hz = config['target_hz']
        target_time_ms = 1000 / target_hz
        
        # Measure actual performance
        input_shape = (batch_size, obs_dim)
        timing_results = measure_single_inference(model, input_shape, device, num_runs=100)
        
        actual_time_ms = timing_results['mean_ms']
        max_achievable_hz = 1000 / actual_time_ms
        
        # Check if target is achievable
        meets_target = actual_time_ms <= target_time_ms
        overhead_ms = max(0, actual_time_ms - target_time_ms)
        
        results[scenario_name] = {
            'description': config['description'],
            'target_hz': target_hz,
            'target_time_ms': target_time_ms,
            'actual_time_ms': actual_time_ms,
            'max_achievable_hz': max_achievable_hz,
            'meets_target': meets_target,
            'overhead_ms': overhead_ms,
            'batch_size': batch_size
        }
    
    return results


def compare_models(obs_dim=8, action_dim=2, device_name='cpu'):
    """Comprehensive comparison between BitNet and standard models"""
    device = torch.device(device_name)
    
    print(f"\n🚀 Speed Benchmark: BitNet vs Standard Models")
    print(f"Device: {device}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print("=" * 60)
    
    # Create models
    config_bitnet = BitACTConfig(
        action_dim=action_dim,
        chunk_size=16,
        use_bitnet=True,
        dim_model=256,
        n_encoder_layers=3,
        n_decoder_layers=4,
    )
    
    config_standard = BitACTConfig(
        action_dim=action_dim,
        chunk_size=16,
        use_bitnet=False,
        dim_model=256,
        n_encoder_layers=3,
        n_decoder_layers=4,
    )
    
    bitnet_model = BitACTPolicy(config_bitnet, observation_dim=obs_dim).to(device)
    standard_model = BitACTPolicy(config_standard, observation_dim=obs_dim).to(device)
    
    models = {
        'BitNet': bitnet_model,
        'Standard': standard_model
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n⚡ Benchmarking {model_name} Model...")
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")
        
        model_results = {}
        
        # 1. Single inference latency
        print("  📊 Single inference latency...")
        single_perf = measure_single_inference(model, (1, obs_dim), device)
        model_results['single_inference'] = single_perf
        print(f"    Mean: {single_perf['mean_ms']:.2f}ms ± {single_perf['std_ms']:.2f}ms")
        print(f"    P95: {single_perf['p95_ms']:.2f}ms, P99: {single_perf['p99_ms']:.2f}ms")
        
        # 2. Batch throughput
        print("  📈 Batch throughput...")
        batch_perf = measure_batch_throughput(model, obs_dim, device)
        model_results['batch_throughput'] = batch_perf
        
        for batch_size, perf in batch_perf.items():
            print(f"    Batch {batch_size}: {perf['throughput_sps']:.1f} samples/sec")
        
        # 3. Memory bandwidth (GPU only)
        if device.type == 'cuda':
            print("  💾 Memory bandwidth...")
            memory_perf = measure_memory_bandwidth(model, obs_dim, device)
            model_results['memory_bandwidth'] = memory_perf
            print(f"    {memory_perf['inferences_per_second']:.1f} inferences/sec")
            print(f"    {memory_perf['memory_per_inference_mb']:.2f} MB/inference")
        
        # 4. Robotics scenarios
        print("  🤖 Robotics scenarios...")
        robotics_perf = robotics_scenario_benchmark(model, obs_dim, device)
        model_results['robotics_scenarios'] = robotics_perf
        
        for scenario, perf in robotics_perf.items():
            status = "✅" if perf['meets_target'] else "❌"
            print(f"    {scenario}: {perf['actual_time_ms']:.1f}ms {status}")
        
        results[model_name] = model_results
    
    # Comparative analysis
    print(f"\n📊 Comparative Analysis")
    print("=" * 40)
    
    bitnet_single = results['BitNet']['single_inference']['mean_ms']
    standard_single = results['Standard']['single_inference']['mean_ms']
    speedup = standard_single / bitnet_single
    
    print(f"Single Inference Speedup: {speedup:.2f}x")
    print(f"BitNet: {bitnet_single:.2f}ms vs Standard: {standard_single:.2f}ms")
    
    # Batch throughput comparison
    print(f"\nBatch Throughput Comparison:")
    for batch_size in [1, 8, 32]:
        if batch_size in results['BitNet']['batch_throughput']:
            bitnet_throughput = results['BitNet']['batch_throughput'][batch_size]['throughput_sps']
            standard_throughput = results['Standard']['batch_throughput'][batch_size]['throughput_sps']
            throughput_speedup = bitnet_throughput / standard_throughput
            print(f"  Batch {batch_size}: {throughput_speedup:.2f}x speedup ({bitnet_throughput:.1f} vs {standard_throughput:.1f} sps)")
    
    # Robotics scenario comparison
    print(f"\nRobotics Scenario Success Rate:")
    scenarios = results['BitNet']['robotics_scenarios'].keys()
    for scenario in scenarios:
        bitnet_success = results['BitNet']['robotics_scenarios'][scenario]['meets_target']
        standard_success = results['Standard']['robotics_scenarios'][scenario]['meets_target']
        print(f"  {scenario}: BitNet {'✅' if bitnet_success else '❌'} vs Standard {'✅' if standard_success else '❌'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Speed benchmark for 1bit_vla models')
    parser.add_argument('--obs_dim', type=int, default=8, help='Observation dimension')
    parser.add_argument('--action_dim', type=int, default=2, help='Action dimension')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device for benchmarking')
    parser.add_argument('--save_results', action='store_true', help='Save results to JSON file')
    parser.add_argument('--output_file', type=str, default='speed_benchmark_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    # Run comprehensive benchmark
    results = compare_models(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        device_name=args.device
    )
    
    # Save results if requested
    if args.save_results:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(results)
        
        with open(args.output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to {args.output_file}")
    
    # Summary recommendations
    print(f"\n🎯 Performance Recommendations")
    print("=" * 35)
    
    bitnet_single = results['BitNet']['single_inference']['mean_ms']
    standard_single = results['Standard']['single_inference']['mean_ms']
    
    if bitnet_single < standard_single:
        speedup = standard_single / bitnet_single
        print(f"✅ BitNet is {speedup:.2f}x faster for inference!")
        print(f"   Recommended for real-time robotics applications.")
    else:
        slowdown = bitnet_single / standard_single
        print(f"⚠️  BitNet is {slowdown:.2f}x slower for inference.")
        print(f"   Consider standard model for latency-critical applications.")
    
    # Check real-time scenarios
    real_time_scenarios = ['real_time_control', 'high_frequency']
    bitnet_rt_success = sum(results['BitNet']['robotics_scenarios'][s]['meets_target'] for s in real_time_scenarios)
    standard_rt_success = sum(results['Standard']['robotics_scenarios'][s]['meets_target'] for s in real_time_scenarios)
    
    print(f"\nReal-time Control Capability:")
    print(f"  BitNet: {bitnet_rt_success}/{len(real_time_scenarios)} scenarios ✅")
    print(f"  Standard: {standard_rt_success}/{len(real_time_scenarios)} scenarios ✅")
    
    if args.device == 'cuda':
        bitnet_memory = results['BitNet']['memory_bandwidth']['memory_per_inference_mb']
        standard_memory = results['Standard']['memory_bandwidth']['memory_per_inference_mb']
        memory_efficiency = standard_memory / bitnet_memory
        print(f"\nMemory Efficiency: {memory_efficiency:.2f}x better with BitNet")
    
    print(f"\n🚀 Speed benchmarking complete!")


if __name__ == "__main__":
    main() 