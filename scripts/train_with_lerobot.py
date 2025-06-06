#!/usr/bin/env python3
"""
Train 1bit_vla models with LeRobot datasets and compare with standard VLA approaches.

This script provides:
1. Training with LeRobot datasets using 1bit_vla models
2. Comparison with standard (non-quantized) VLA models 
3. Comprehensive evaluation and analysis
4. Support for multiple datasets and model configurations

Usage:
    # Train BitACT on ALOHA dataset
    python scripts/train_with_lerobot.py --dataset lerobot/aloha_static_coffee --model bitact --use_bitnet

    # Train standard ACT for comparison
    python scripts/train_with_lerobot.py --dataset lerobot/aloha_static_coffee --model bitact --no-use_bitnet

    # Train on PushT with diffusion-style model
    python scripts/train_with_lerobot.py --dataset lerobot/pusht --model vla_bitnet --use_bitnet
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# LeRobot imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "lerobot"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType

# 1bit_vla imports  
sys.path.append(str(Path(__file__).parent.parent / "src"))
from bit_vla import (
    VLABitNet, BitACTPolicy, BitACTConfig, VLATrainer, 
    print_model_info, analyze_quantization
)


def setup_logging(output_dir: Path) -> None:
    """Setup logging to file and console."""
    log_file = output_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get dataset metadata and suggested configurations."""
    dataset_configs = {
        "lerobot/pusht": {
            "action_dim": 2,
            "chunk_size": 16, 
            "observation_keys": ["observation.image", "observation.state"],
            "suggested_lr": 1e-4,
            "batch_size": 64,
            "train_steps": 5000,
        },
        "lerobot/aloha_static_coffee": {
            "action_dim": 14,  # 7 per arm
            "chunk_size": 50,
            "observation_keys": ["observation.images.cam_high", "observation.state"],
            "suggested_lr": 1e-4, 
            "batch_size": 32,
            "train_steps": 10000,
        },
        "lerobot/aloha_sim_insertion_human": {
            "action_dim": 14,
            "chunk_size": 50,
            "observation_keys": ["observation.images.top", "observation.state"],
            "suggested_lr": 1e-4,
            "batch_size": 32, 
            "train_steps": 8000,
        },
        "lerobot/xarm_lift_medium": {
            "action_dim": 7,
            "chunk_size": 32,
            "observation_keys": ["observation.images.wrist", "observation.state"],
            "suggested_lr": 1e-4,
            "batch_size": 64,
            "train_steps": 6000,
        },
    }
    
    # Default config for unknown datasets
    default_config = {
        "action_dim": 7,
        "chunk_size": 32,
        "observation_keys": ["observation.image", "observation.state"],
        "suggested_lr": 1e-4,
        "batch_size": 32,
        "train_steps": 5000,
    }
    
    return dataset_configs.get(dataset_name, default_config)


def create_model(model_type: str, dataset_info: Dict, use_bitnet: bool = True) -> nn.Module:
    """Create model based on type and dataset info."""
    
    if model_type == "bitact":
        config = BitACTConfig(
            action_dim=dataset_info["action_dim"],
            chunk_size=dataset_info["chunk_size"],
            use_bitnet=use_bitnet,
            n_action_steps=dataset_info["chunk_size"],
        )
        
        # Estimate observation dimension (will be adjusted based on actual data)
        obs_dim = 32  # Default, will be updated during data loading
        model = BitACTPolicy(config, observation_dim=obs_dim)
        
    elif model_type == "vla_bitnet":
        # For more complex multimodal tasks
        model = VLABitNet(
            hidden_dim=512,
            action_dim=dataset_info["action_dim"],
            use_bitnet=use_bitnet,
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def prepare_dataset(dataset_name: str, dataset_info: Dict) -> tuple[LeRobotDataset, Dict]:
    """Prepare LeRobot dataset with appropriate delta timestamps."""
    
    # Get dataset metadata
    metadata = LeRobotDatasetMetadata(dataset_name)
    
    # Configure delta timestamps based on dataset type
    if "pusht" in dataset_name:
        delta_timestamps = {
            "observation.image": [-0.1, 0.0],
            "observation.state": [-0.1, 0.0], 
            "action": [i * 0.1 for i in range(dataset_info["chunk_size"])],
        }
    elif "aloha" in dataset_name:
        delta_timestamps = {
            "observation.images.cam_high": [0.0],
            "observation.state": [0.0],
            "action": [i / metadata.fps for i in range(dataset_info["chunk_size"])],
        }
    else:
        # Generic configuration
        fps = metadata.fps
        chunk_size = dataset_info["chunk_size"]
        delta_timestamps = {
            key: [0.0] for key in dataset_info["observation_keys"]
        }
        delta_timestamps["action"] = [i / fps for i in range(chunk_size)]
    
    # Load dataset
    dataset = LeRobotDataset(
        dataset_name,
        delta_timestamps=delta_timestamps,
        video_backend="pyav"  # Use PyAV for better compatibility
    )
    
    # Prepare features for policy creation  
    features = dataset_to_policy_features(metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    dataset_stats = {
        "num_episodes": dataset.num_episodes,
        "num_frames": dataset.num_frames,
        "fps": metadata.fps,
        "features": features,
        "input_features": input_features,
        "output_features": output_features,
    }
    
    return dataset, dataset_stats


def create_data_collator(model_type: str):
    """Create appropriate data collation function."""
    
    def collate_fn(batch):
        """Custom collate function for different model types."""
        
        # Standard tensor collation
        collated = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = [item[key] for item in batch]
        
        return collated
    
    return collate_fn


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str,
    epoch: int,
    log_freq: int = 100,
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
        
        try:
            # Forward pass - adapt based on model type
            if model_type == "bitact":
                # For BitACT, we need observations and actions
                if "observation.state" in batch:
                    observations = batch["observation.state"]
                    if observations.dim() > 2:
                        observations = observations.flatten(1)  # Flatten extra dims
                else:
                    # Use first available observation
                    obs_key = next(k for k in batch.keys() if k.startswith("observation"))
                    observations = batch[obs_key]
                    if observations.dim() > 2:
                        observations = observations.flatten(1)
                
                # Get actions
                actions = batch["action"]
                if actions.dim() > 2:
                    actions = actions.flatten(1)
                
                # BitACT forward
                predicted_actions = model(observations)
                
                # Compute loss
                loss = nn.MSELoss()(predicted_actions, actions)
                
            elif model_type == "vla_bitnet":
                # For VLA models, handle images and language
                images = None
                tokens = None
                
                # Extract images
                for key in batch.keys():
                    if "image" in key and isinstance(batch[key], torch.Tensor):
                        images = batch[key]
                        if images.dim() == 5:  # [batch, seq, c, h, w]
                            images = images[:, -1]  # Take last frame
                        break
                
                # Create dummy tokens if needed
                if tokens is None:
                    batch_size = images.shape[0] if images is not None else len(batch["action"])
                    tokens = torch.zeros(batch_size, 32, dtype=torch.long, device=device)
                
                # Get actions
                actions = batch["action"]
                if actions.dim() > 2:
                    actions = actions[:, -1]  # Take last action
                
                # VLA forward
                predicted_actions = model(images, tokens)
                
                # Compute loss
                loss = nn.MSELoss()(predicted_actions, actions)
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log frequently
            if batch_idx % log_freq == 0:
                logging.info(
                    f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}"
                )
                
                if wandb.run is not None:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx,
                    })
        
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            logging.error(f"Batch shapes: {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in batch.items()]}")
            raise e
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        "avg_loss": avg_loss,
        "num_batches": num_batches,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
) -> Dict[str, float]:
    """Evaluate model performance."""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }
            
            try:
                # Same forward logic as training
                if model_type == "bitact":
                    if "observation.state" in batch:
                        observations = batch["observation.state"]
                        if observations.dim() > 2:
                            observations = observations.flatten(1)
                    else:
                        obs_key = next(k for k in batch.keys() if k.startswith("observation"))
                        observations = batch[obs_key]
                        if observations.dim() > 2:
                            observations = observations.flatten(1)
                    
                    actions = batch["action"]
                    if actions.dim() > 2:
                        actions = actions.flatten(1)
                    
                    predicted_actions = model(observations)
                    loss = nn.MSELoss()(predicted_actions, actions)
                    
                elif model_type == "vla_bitnet":
                    images = None
                    for key in batch.keys():
                        if "image" in key and isinstance(batch[key], torch.Tensor):
                            images = batch[key]
                            if images.dim() == 5:
                                images = images[:, -1]
                            break
                    
                    if images is None:
                        continue
                        
                    batch_size = images.shape[0]
                    tokens = torch.zeros(batch_size, 32, dtype=torch.long, device=device)
                    
                    actions = batch["action"]
                    if actions.dim() > 2:
                        actions = actions[:, -1]
                    
                    predicted_actions = model(images, tokens)
                    loss = nn.MSELoss()(predicted_actions, actions)
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logging.warning(f"Evaluation error: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    return {
        "eval_loss": avg_loss,
        "num_eval_batches": num_batches,
    }


def run_comparison_experiment(
    dataset_name: str,
    model_type: str, 
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run training experiment and comparison."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Get dataset info and prepare dataset
    dataset_info = get_dataset_info(dataset_name)
    logging.info(f"Dataset info: {dataset_info}")
    
    dataset, dataset_stats = prepare_dataset(dataset_name, dataset_info)
    logging.info(f"Dataset loaded: {dataset_stats['num_episodes']} episodes, {dataset_stats['num_frames']} frames")
    
    # Update dataset info with actual observation dimension
    sample_batch = dataset[0]
    if "observation.state" in sample_batch:
        obs_tensor = sample_batch["observation.state"]
        if obs_tensor.dim() > 1:
            obs_dim = obs_tensor.flatten().shape[0]
        else:
            obs_dim = obs_tensor.shape[0]
    else:
        obs_dim = 32  # Default
    
    dataset_info["obs_dim"] = obs_dim
    logging.info(f"Observation dimension: {obs_dim}")
    
    # Create models for comparison
    results = {}
    
    # Train BitNet version if requested
    if args.use_bitnet:
        logging.info("Training BitNet model...")
        bitnet_model = create_model(model_type, dataset_info, use_bitnet=True)
        
        # Update observation dimension for BitACT
        if model_type == "bitact" and hasattr(bitnet_model, 'obs_projection'):
            bitnet_model.obs_projection = nn.Linear(obs_dim, bitnet_model.config.dim_model)
        
        bitnet_model.to(device)
        
        print_model_info(bitnet_model, "BitNet Model")
        
        # Training
        bitnet_results = train_model(
            bitnet_model, dataset, device, model_type, 
            output_dir / "bitnet", dataset_info, args, "bitnet"
        )
        results["bitnet"] = bitnet_results
    
    # Train standard version for comparison
    if args.compare_standard:
        logging.info("Training standard (FP32) model for comparison...")
        standard_model = create_model(model_type, dataset_info, use_bitnet=False)
        
        # Update observation dimension  
        if model_type == "bitact" and hasattr(standard_model, 'obs_projection'):
            standard_model.obs_projection = nn.Linear(obs_dim, standard_model.config.dim_model)
        
        standard_model.to(device)
        
        print_model_info(standard_model, "Standard Model")
        
        # Training
        standard_results = train_model(
            standard_model, dataset, device, model_type,
            output_dir / "standard", dataset_info, args, "standard"
        )
        results["standard"] = standard_results
    
    # Generate comparison report
    if len(results) > 1:
        comparison_report = generate_comparison_report(results, output_dir)
        logging.info("Comparison report generated!")
        
    return results


def train_model(
    model: nn.Module,
    dataset: LeRobotDataset, 
    device: torch.device,
    model_type: str,
    output_dir: Path,
    dataset_info: Dict,
    args: argparse.Namespace,
    model_name: str,
) -> Dict[str, Any]:
    """Train a single model."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data loading
    collate_fn = create_data_collator(model_type)
    
    # Split dataset for train/validation
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size or dataset_info["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size or dataset_info["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    
    # Optimizer setup
    if hasattr(model, 'get_optim_params') and args.use_bitnet:
        # Use BitNet-specific optimizer settings
        param_groups = model.get_optim_params()
        optimizer = torch.optim.AdamW(param_groups)
    else:
        # Standard optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate or dataset_info["suggested_lr"],
            weight_decay=1e-5,
        )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    logging.info(f"Starting training for {model_name}...")
    
    for epoch in range(args.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, model_type, epoch
        )
        train_losses.append(train_metrics["avg_loss"])
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, device, model_type)
        val_losses.append(val_metrics["eval_loss"])
        
        # Log metrics
        logging.info(
            f"Epoch {epoch}: Train Loss = {train_metrics['avg_loss']:.4f}, "
            f"Val Loss = {val_metrics['eval_loss']:.4f}"
        )
        
        if wandb.run is not None:
            wandb.log({
                f"{model_name}/train_loss": train_metrics["avg_loss"],
                f"{model_name}/val_loss": val_metrics["eval_loss"],
                f"{model_name}/epoch": epoch,
            })
        
        # Save best model
        if val_metrics["eval_loss"] < best_val_loss:
            best_val_loss = val_metrics["eval_loss"]
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
                "train_loss": train_metrics["avg_loss"],
            }, checkpoint_path)
            logging.info(f"Saved best model with val_loss: {best_val_loss:.4f}")
    
    # Final evaluation and analysis
    final_results = {
        "model_name": model_name,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    # BitNet-specific analysis
    if hasattr(model, 'get_quantization_summary'):
        quant_summary = model.get_quantization_summary()
        final_results.update(quant_summary)
        
        # Detailed quantization analysis
        quant_analysis = analyze_quantization(model, step=args.num_epochs, loss_history=train_losses)
        final_results["quantization_analysis"] = quant_analysis
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    return final_results


def generate_comparison_report(results: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Generate comprehensive comparison report."""
    
    report = {
        "summary": {},
        "detailed_comparison": {},
        "recommendations": [],
    }
    
    # Summary comparison
    for name, result in results.items():
        report["summary"][name] = {
            "best_val_loss": result["best_val_loss"],
            "num_parameters": result["num_parameters"],
            "memory_usage": result.get("memory_savings", {}).get("absolute_mb", "N/A"),
        }
    
    # Performance comparison
    if "bitnet" in results and "standard" in results:
        bitnet_loss = results["bitnet"]["best_val_loss"]
        standard_loss = results["standard"]["best_val_loss"]
        
        performance_gap = ((bitnet_loss - standard_loss) / standard_loss) * 100
        
        report["detailed_comparison"]["performance_gap_percent"] = performance_gap
        
        # Memory comparison
        if "memory_savings" in results["bitnet"]:
            memory_savings = results["bitnet"]["memory_savings"]
            report["detailed_comparison"]["memory_savings"] = memory_savings
        
        # Generate recommendations
        if performance_gap < 5:  # Less than 5% performance loss
            report["recommendations"].append(
                "BitNet quantization is highly recommended - minimal performance loss with significant memory savings"
            )
        elif performance_gap < 15:
            report["recommendations"].append(
                "BitNet quantization is viable - moderate performance trade-off for memory benefits"
            )
        else:
            report["recommendations"].append(
                "Consider fine-tuning BitNet model or using progressive quantization"
            )
    
    # Save report
    report_path = output_dir / "comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    for name, summary in report["summary"].items():
        print(f"{name.upper()}: Loss={summary['best_val_loss']:.4f}, Params={summary['num_parameters']:,}")
    
    if "performance_gap_percent" in report["detailed_comparison"]:
        gap = report["detailed_comparison"]["performance_gap_percent"]
        print(f"\nPerformance Gap: {gap:+.1f}%")
    
    print("\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"- {rec}")
    print("="*50)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Train 1bit_vla models with LeRobot datasets"
    )
    
    # Dataset and model
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="lerobot/pusht",
        help="LeRobot dataset to use (e.g., lerobot/pusht, lerobot/aloha_static_coffee)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["bitact", "vla_bitnet"],
        default="bitact",
        help="Model type to train"
    )
    
    # Training configuration
    parser.add_argument("--use_bitnet", action="store_true", help="Use BitNet quantization")
    parser.add_argument("--compare_standard", action="store_true", help="Also train standard model for comparison")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size (default: auto-detect)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (default: auto-detect)")
    
    # Output and logging
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/lerobot_training",
        help="Output directory for results"
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="1bit_vla_lerobot", help="W&B project name")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir) / f"{args.dataset.replace('/', '_')}_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.dataset}_{args.model}",
            config=vars(args),
        )
    
    # Log configuration
    logging.info(f"Configuration: {vars(args)}")
    logging.info(f"Output directory: {output_dir}")
    
    try:
        # Run experiment
        results = run_comparison_experiment(
            args.dataset, args.model, output_dir, args
        )
        
        logging.info("Training completed successfully!")
        logging.info(f"Results saved to: {output_dir}")
        
        # Print final summary
        print(f"\nTraining completed! Results in: {output_dir}")
        
        if wandb.run is not None:
            wandb.finish()
            
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 