#!/usr/bin/env python3
"""
Simple example showing how to use 1bit_vla models with LeRobot datasets.

This example demonstrates:
1. Loading LeRobot datasets
2. Training BitACT policies on real robotics data
3. Comparing BitNet vs standard models
4. Basic evaluation and analysis

Requirements:
- lerobot installed in parent directory 
- 1bit_vla package installed
- torch, wandb (optional)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import time

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "lerobot"))
sys.path.append(str(Path(__file__).parent.parent / "src"))

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# 1bit_vla imports
from bit_vla import BitACTPolicy, BitACTConfig, print_model_info, analyze_quantization


def main():
    print("🤖 1bit_vla + LeRobot Integration Example")
    print("=" * 50)
    
    # Configuration
    dataset_name = "lerobot/pusht"  # Start with a simple dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load LeRobot dataset
    print("\n📊 Loading LeRobot dataset...")
    
    # Configure temporal relationships for action chunking
    delta_timestamps = {
        "observation.image": [-0.1, 0.0],  # Previous and current image
        "observation.state": [-0.1, 0.0],  # Previous and current state  
        "action": [i * 0.1 for i in range(16)],  # 16 future actions (1.6 seconds)
    }
    
    dataset = LeRobotDataset(dataset_name, delta_timestamps=delta_timestamps, video_backend="pyav")
    metadata = LeRobotDatasetMetadata(dataset_name)
    
    print(f"Dataset: {dataset_name}")
    print(f"Episodes: {dataset.num_episodes}")
    print(f"Frames: {dataset.num_frames}")
    print(f"FPS: {metadata.fps}")
    
    # Inspect a sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Step 2: Create BitACT models for comparison
    print("\n🔧 Creating models...")
    
    # Configuration for PushT task
    config_bitnet = BitACTConfig(
        action_dim=2,  # PushT has 2D actions (x, y)
        chunk_size=16,  # Predict 16 actions into future
        use_bitnet=True,
        dim_model=256,  # Smaller for this example
        n_encoder_layers=3,
        n_decoder_layers=4,
    )
    
    config_standard = BitACTConfig(
        action_dim=2,
        chunk_size=16, 
        use_bitnet=False,  # Standard FP32 model
        dim_model=256,
        n_encoder_layers=3,
        n_decoder_layers=4,
    )
    
    # Get observation dimension from sample
    obs_state = sample["observation.state"]
    obs_dim = obs_state.flatten().shape[0] if obs_state.dim() > 1 else obs_state.shape[0]
    print(f"Observation dimension: {obs_dim}")
    
    # Create models
    bitnet_model = BitACTPolicy(config_bitnet, observation_dim=obs_dim).to(device)
    standard_model = BitACTPolicy(config_standard, observation_dim=obs_dim).to(device)
    
    print_model_info(bitnet_model, "BitNet Model")
    print_model_info(standard_model, "Standard Model")
    
    # Step 3: Simple training loop
    print("\n🏃 Training models...")
    
    # Create data loader
    def collate_fn(batch):
        collated = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = [item[key] for item in batch]
        return collated
    
    # Use small subset for quick example
    subset_size = min(1000, len(dataset))
    subset_indices = torch.randperm(len(dataset))[:subset_size].tolist()
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    dataloader = DataLoader(
        subset_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues in example
    )
    
    # Training function
    def train_model(model, model_name, epochs=3):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        losses = []
        
        print(f"\nTraining {model_name}...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Get observations and actions
                observations = batch["observation.state"]
                if observations.dim() > 2:
                    observations = observations.flatten(1)
                
                actions = batch["action"]
                # Ensure actions are properly shaped for the model
                if actions.dim() == 3:  # [batch, chunk_size, action_dim]
                    actions = actions.flatten(1)  # [batch, chunk_size * action_dim]
                elif actions.dim() == 2 and actions.shape[1] == 2:  # Single action
                    # Repeat single action to match chunk size
                    actions = actions.unsqueeze(1).repeat(1, 16, 1).flatten(1)
                
                # Forward pass
                predicted_actions = model(observations)
                
                # Ensure predicted actions match target shape
                if predicted_actions.shape != actions.shape:
                    if predicted_actions.dim() == 3:
                        predicted_actions = predicted_actions.flatten(1)
                
                loss = nn.MSELoss()(predicted_actions, actions)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        
        return losses
    
    # Train both models
    bitnet_losses = train_model(bitnet_model, "BitNet")
    standard_losses = train_model(standard_model, "Standard")
    
    # Step 4: Analysis and comparison
    print("\n📈 Analysis and Comparison")
    print("=" * 30)
    
    # Performance comparison
    final_bitnet_loss = bitnet_losses[-1]
    final_standard_loss = standard_losses[-1]
    performance_gap = ((final_bitnet_loss - final_standard_loss) / final_standard_loss) * 100
    
    print(f"Final BitNet Loss: {final_bitnet_loss:.4f}")
    print(f"Final Standard Loss: {final_standard_loss:.4f}")
    print(f"Performance Gap: {performance_gap:+.1f}%")
    
    # Memory analysis
    bitnet_params = sum(p.numel() for p in bitnet_model.parameters())
    standard_params = sum(p.numel() for p in standard_model.parameters())
    
    print(f"\nModel Sizes:")
    print(f"BitNet Parameters: {bitnet_params:,}")
    print(f"Standard Parameters: {standard_params:,}")
    
    # Speed Benchmarking
    print(f"\n⚡ Speed Benchmarking")
    print("=" * 20)
    
    # Single inference speed test
    def measure_inference_speed(model, input_shape, num_runs=100):
        model.eval()
        times = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                dummy_input = torch.randn(input_shape, device=device)
                _ = model(dummy_input)
        
        # Actual timing
        with torch.no_grad():
            for _ in range(num_runs):
                dummy_input = torch.randn(input_shape, device=device)
                
                start_time = time.perf_counter()
                _ = model(dummy_input)
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'std_ms': (sum([(t - sum(times)/len(times))**2 for t in times]) / len(times))**0.5
        }
    
    # Test single inference
    single_shape = (1, obs_dim)
    bitnet_single = measure_inference_speed(bitnet_model, single_shape)
    standard_single = measure_inference_speed(standard_model, single_shape)
    
    single_speedup = standard_single['mean_ms'] / bitnet_single['mean_ms']
    
    print(f"Single Inference Speed:")
    print(f"  BitNet: {bitnet_single['mean_ms']:.2f}ms ± {bitnet_single['std_ms']:.2f}ms")
    print(f"  Standard: {standard_single['mean_ms']:.2f}ms ± {standard_single['std_ms']:.2f}ms")
    print(f"  Speedup: {single_speedup:.2f}x {'✅' if single_speedup > 1.0 else '⚠️'}")
    
    # Test batch inference
    batch_shape = (32, obs_dim)
    bitnet_batch = measure_inference_speed(bitnet_model, batch_shape, num_runs=50)
    standard_batch = measure_inference_speed(standard_model, batch_shape, num_runs=50)
    
    # Calculate throughput (samples per second)
    bitnet_throughput = 32 / (bitnet_batch['mean_ms'] / 1000)
    standard_throughput = 32 / (standard_batch['mean_ms'] / 1000)
    throughput_speedup = bitnet_throughput / standard_throughput
    
    print(f"\nBatch Inference (32 samples):")
    print(f"  BitNet: {bitnet_batch['mean_ms']:.2f}ms ({bitnet_throughput:.1f} samples/sec)")
    print(f"  Standard: {standard_batch['mean_ms']:.2f}ms ({standard_throughput:.1f} samples/sec)")
    print(f"  Throughput Speedup: {throughput_speedup:.2f}x {'✅' if throughput_speedup > 1.0 else '⚠️'}")
    
    # Real-time robotics scenarios
    print(f"\nReal-time Robotics Scenarios:")
    scenarios = {
        '10Hz Control Loop': 100,    # 100ms target
        '20Hz Control Loop': 50,     # 50ms target
        '50Hz Control Loop': 20,     # 20ms target
    }
    
    for scenario_name, target_ms in scenarios.items():
        bitnet_capable = bitnet_single['mean_ms'] <= target_ms
        standard_capable = standard_single['mean_ms'] <= target_ms
        
        print(f"  {scenario_name} ({target_ms}ms):")
        print(f"    BitNet: {'✅ Capable' if bitnet_capable else '❌ Too slow'} ({bitnet_single['mean_ms']:.1f}ms)")
        print(f"    Standard: {'✅ Capable' if standard_capable else '❌ Too slow'} ({standard_single['mean_ms']:.1f}ms)")
    
    # BitNet-specific analysis
    if hasattr(bitnet_model, 'get_quantization_summary'):
        quant_summary = bitnet_model.get_quantization_summary()
        print(f"\nQuantization Summary:")
        print(f"Memory Savings: {quant_summary.get('memory_savings', {}).get('percentage', 'N/A')}%")
        
        _ = analyze_quantization(bitnet_model, step=len(bitnet_losses), loss_history=bitnet_losses)
        print("Quantization analysis completed!")
    
    # Step 5: Save results
    print("\n💾 Saving results...")
    
    results = {
        "dataset": dataset_name,
        "dataset_info": {
            "episodes": dataset.num_episodes,
            "frames": dataset.num_frames,
            "fps": metadata.fps,
        },
        "training_results": {
            "bitnet_losses": bitnet_losses,
            "standard_losses": standard_losses,
            "performance_gap_percent": performance_gap,
        },
        "model_comparison": {
            "bitnet_params": bitnet_params,
            "standard_params": standard_params,
        }
    }
    
    output_file = Path("lerobot_integration_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Step 6: Recommendations
    print("\n🎯 Recommendations")
    print("=" * 20)
    
    if performance_gap < 5:
        print("✅ BitNet quantization is highly recommended!")
        print("   Minimal performance loss with significant memory savings.")
    elif performance_gap < 15:
        print("⚖️  BitNet quantization is viable.")
        print("   Moderate performance trade-off for memory benefits.")
    else:
        print("⚠️  Consider fine-tuning or longer training for BitNet.")
        print("   Performance gap is significant.")
    
    print(f"\n📚 Try other datasets:")
    suggested_datasets = [
        "lerobot/aloha_static_coffee",
        "lerobot/aloha_sim_insertion_human", 
        "lerobot/xarm_lift_medium",
        "lerobot/pusht_image",
    ]
    for dataset_suggestion in suggested_datasets:
        print(f"   - {dataset_suggestion}")
    
    print("\n✨ Integration complete! BitNet + LeRobot working together!")


if __name__ == "__main__":
    main() 