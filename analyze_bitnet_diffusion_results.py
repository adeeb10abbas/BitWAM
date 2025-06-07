#!/usr/bin/env python3
"""
Analysis of BitNet Compatible Diffusion Policy Results

This script analyzes the performance results from our BitNet compatible
diffusion policy implementation and creates visualizations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results():
    """Load the benchmark results."""
    with open("bitnet_compatible_results.json", "r") as f:
        return json.load(f)


def create_performance_visualization(results):
    """Create comprehensive performance visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BitNet Compatible Diffusion Policy Performance Analysis\n(Batch=1, 20 Diffusion Steps)', 
                 fontsize=16, fontweight='bold')
    
    models = list(results.keys())
    times = [results[model]["avg_diffusion_time_ms"] for model in models]
    throughputs = [results[model]["throughput_samples_per_sec"] for model in models]
    robot_freqs = [results[model]["robot_control_freq_hz"] for model in models]
    
    # Define colors
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. Inference Time Comparison
    bars1 = ax1.bar(models, times, color=colors, alpha=0.8)
    ax1.set_title('Average Inference Time per Sample', fontweight='bold')
    ax1.set_ylabel('Time (ms)')
    ax1.set_ylim(0, max(times) * 1.2)
    
    # Add value labels on bars
    for bar, time in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time:.2f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Add performance target lines
    ax1.axhline(y=10.0, color='green', linestyle='--', alpha=0.7, label='100Hz Target (10ms)')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1kHz Target (1ms)')
    ax1.legend()
    
    # 2. Throughput Comparison
    bars2 = ax2.bar(models, throughputs, color=colors, alpha=0.8)
    ax2.set_title('Inference Throughput', fontweight='bold')
    ax2.set_ylabel('Samples/sec')
    ax2.set_ylim(0, max(throughputs) * 1.2)
    
    for bar, throughput in zip(bars2, throughputs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{throughput:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Robot Control Frequency
    bars3 = ax3.bar(models, robot_freqs, color=colors, alpha=0.8)
    ax3.set_title('Maximum Robot Control Frequency', fontweight='bold')
    ax3.set_ylabel('Control Frequency (Hz)')
    ax3.set_ylim(0, max(robot_freqs) * 1.2)
    
    for bar, freq in zip(bars3, robot_freqs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{freq:.0f}Hz', ha='center', va='bottom', fontweight='bold')
    
    # Add frequency requirement lines
    ax3.axhline(y=100, color='blue', linestyle='--', alpha=0.7, label='Typical Robot (100Hz)')
    ax3.axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='High-freq (1kHz)')
    ax3.legend()
    
    # 4. Relative Performance vs FP32
    baseline_time = results["FP32 Standard"]["avg_diffusion_time_ms"]
    relative_perf = [baseline_time / results[model]["avg_diffusion_time_ms"] for model in models]
    
    bars4 = ax4.bar(models, relative_perf, color=colors, alpha=0.8)
    ax4.set_title('Relative Performance vs FP32 Standard', fontweight='bold')
    ax4.set_ylabel('Relative Performance (higher = better)')
    ax4.set_ylim(0, max(relative_perf) * 1.2)
    ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='FP32 Baseline')
    
    for bar, perf in zip(bars4, relative_perf):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{perf:.2f}×', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bitnet_diffusion_performance.png', dpi=300, bbox_inches='tight')
    print("📊 Performance visualization saved as 'bitnet_diffusion_performance.png'")


def analyze_quantization_overhead():
    """Analyze the quantization overhead in detail."""
    results = load_results()
    
    print("\n" + "="*60)
    print("🔬 DETAILED QUANTIZATION OVERHEAD ANALYSIS")
    print("="*60)
    
    fp32_time = results["FP32 Standard"]["avg_diffusion_time_ms"]
    std_time = results["Compatible Standard"]["avg_diffusion_time_ms"]
    fast_time = results["Compatible Fast"]["avg_diffusion_time_ms"]
    ultra_fast_time = results["Compatible Ultra-Fast"]["avg_diffusion_time_ms"]
    
    print(f"\n📈 Performance Breakdown:")
    print(f"   FP32 Standard:     {fp32_time:.2f}ms (baseline)")
    print(f"   BitNet Standard:   {std_time:.2f}ms ({std_time/fp32_time:.2f}× slower)")
    print(f"   BitNet Fast:       {fast_time:.2f}ms ({fast_time/fp32_time:.2f}× slower)")
    print(f"   BitNet Ultra-Fast: {ultra_fast_time:.2f}ms ({ultra_fast_time/fp32_time:.2f}× slower)")
    
    print(f"\n⚡ Optimization Impact:")
    print(f"   Standard → Fast:       {std_time/fast_time:.2f}× speedup")
    print(f"   Standard → Ultra-Fast: {std_time/ultra_fast_time:.2f}× speedup")
    print(f"   Fast → Ultra-Fast:     {fast_time/ultra_fast_time:.2f}× speedup")
    
    # Calculate overhead components
    base_overhead = std_time - fp32_time
    optimized_overhead = ultra_fast_time - fp32_time
    overhead_reduction = base_overhead - optimized_overhead
    
    print(f"\n🎯 Overhead Analysis:")
    print(f"   Base BitNet overhead:      {base_overhead:.2f}ms")
    print(f"   Optimized BitNet overhead: {optimized_overhead:.2f}ms")
    print(f"   Overhead reduction:        {overhead_reduction:.2f}ms ({overhead_reduction/base_overhead*100:.1f}%)")
    
    # Robotics viability analysis
    print(f"\n🤖 Robotics Control Viability:")
    for model, time_ms in [("FP32", fp32_time), ("BitNet Ultra-Fast", ultra_fast_time)]:
        max_freq = 1000 / time_ms
        print(f"   {model}:")
        print(f"     Max control freq: {max_freq:.0f}Hz")
        print(f"     Suitable for ≤100Hz: {'✅' if max_freq >= 100 else '❌'}")
        print(f"     Suitable for ≤500Hz: {'✅' if max_freq >= 500 else '❌'}")
        print(f"     Suitable for ≤1kHz:  {'✅' if max_freq >= 1000 else '❌'}")


def analyze_diffusion_scaling():
    """Analyze how diffusion steps affect performance."""
    print(f"\n🔄 Diffusion Process Analysis:")
    print("-" * 40)
    
    # Each sample requires 20 forward passes (diffusion steps)
    diffusion_steps = 20
    results = load_results()
    
    for model_name, metrics in results.items():
        total_time = metrics["avg_diffusion_time_ms"]
        time_per_step = total_time / diffusion_steps
        forward_calls = metrics["forward_calls_per_sample"]
        
        print(f"\n{model_name}:")
        print(f"   Total diffusion time: {total_time:.2f}ms")
        print(f"   Time per step:        {time_per_step:.3f}ms")
        print(f"   Forward calls:        {forward_calls:.0f}")
        print(f"   Time per forward:     {total_time/forward_calls:.3f}ms")


def generate_summary_report():
    """Generate a comprehensive summary report."""
    results = load_results()
    
    print(f"\n📋 EXECUTIVE SUMMARY REPORT")
    print("="*60)
    
    fp32_time = results["FP32 Standard"]["avg_diffusion_time_ms"]
    best_bitnet_time = results["Compatible Ultra-Fast"]["avg_diffusion_time_ms"]
    
    print(f"\n🎯 Key Findings:")
    print(f"   • FP32 remains fastest at {fp32_time:.2f}ms per sample")
    print(f"   • Best BitNet: {best_bitnet_time:.2f}ms ({best_bitnet_time/fp32_time:.2f}× slower)")
    print(f"   • BitNet optimizations provide {results['Compatible Standard']['avg_diffusion_time_ms']/best_bitnet_time:.2f}× improvement")
    print(f"   • All models suitable for ≤100Hz robot control")
    print(f"   • Only FP32 suitable for 1kHz control loops")
    
    print(f"\n💡 Technical Insights:")
    print(f"   • Quantization overhead dominates at batch=1")
    print(f"   • Diffusion requires 20 forward passes per sample")
    print(f"   • GPU hardware optimized for FP32, not 1-bit ops")
    print(f"   • Memory bandwidth not the bottleneck at batch=1")
    
    print(f"\n🚀 Recommendations:")
    print(f"   1. Use FP32 for maximum performance")
    print(f"   2. Use BitNet Ultra-Fast if memory constrained")
    print(f"   3. Consider larger batch sizes for BitNet benefits")
    print(f"   4. BitNet more suitable for training or inference at scale")


def main():
    """Main analysis function."""
    print("🔍 BitNet Compatible Diffusion Policy Results Analysis")
    
    # Load results
    results = load_results()
    
    # Create visualization
    create_performance_visualization(results)
    
    # Detailed analysis
    analyze_quantization_overhead()
    analyze_diffusion_scaling()
    generate_summary_report()
    
    print(f"\n✅ Analysis complete! Check 'bitnet_diffusion_performance.png' for visualization.")


if __name__ == "__main__":
    main() 