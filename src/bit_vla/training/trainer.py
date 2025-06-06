"""
VLA Trainer for 1-bit models with BitNet quantization.

This module provides a comprehensive training framework for VLA models with
integrated support for BitNet quantization, logging, and evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any
import time
import os
from pathlib import Path

from .bitnet_optimizer import BitNetOptimizer
from ..utils.model_analysis import print_model_info, analyze_quantization


class VLATrainer:
    """
    Comprehensive trainer for VLA models with BitNet support.
    
    This trainer handles the complete training loop including BitNet optimization,
    quantization analysis, logging, and checkpointing.
    
    Args:
        model: VLA model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        device: Device to train on
        output_dir: Directory to save checkpoints and logs
        
    Example:
        >>> model = VLABitNet(hidden_dim=512, action_dim=7)
        >>> trainer = VLATrainer(model, train_loader, device="cuda")
        >>> trainer.train(num_epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
        output_dir: str = "outputs/vla_training",
        log_freq: int = 100,
        save_freq: int = 1000,
        eval_freq: int = 500
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.step = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.lr_history = []
        
        # Print model info
        print_model_info(self.model, "VLA Model")
        
    def setup_optimizer(
        self,
        stage1_lr: float = 1e-3,
        stage2_lr: float = 1e-4,
        stage1_steps: int = 6000,
        weight_decay: float = 0.1,
        **kwargs
    ) -> None:
        """Setup BitNet optimizer with two-stage learning rate schedule."""
        self.optimizer = BitNetOptimizer(
            self.model,
            stage1_lr=stage1_lr,
            stage2_lr=stage2_lr,
            stage1_steps=stage1_steps,
            weight_decay=weight_decay,
            **kwargs
        )
        print("✅ BitNet optimizer configured")
        
    def setup_loss_function(self, loss_fn: Optional[Callable] = None) -> None:
        """Setup loss function for training."""
        if loss_fn is None:
            # Default to MSE loss for action prediction
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn
        print(f"✅ Loss function: {type(self.loss_fn).__name__}")
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform a single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Update learning rate schedule
        if hasattr(self, 'optimizer'):
            self.optimizer.step_lr_schedule(self.step)
        
        # Forward pass
        start_time = time.time()
        
        if "images" in batch and "token_ids" in batch:
            # VLA model forward pass
            predictions = self.model(
                batch["images"], 
                batch["token_ids"],
                batch.get("attention_mask", None)
            )
        elif "observations" in batch:
            # Policy model forward pass
            predictions = self.model(batch["observations"])
        else:
            raise ValueError("Batch must contain either (images, token_ids) or observations")
        
        # Compute loss
        targets = batch.get("actions", batch.get("targets"))
        if targets is None:
            raise ValueError("Batch must contain actions or targets")
            
        loss = self.loss_fn(predictions, targets)
        forward_time = time.time() - start_time
        
        # Backward pass and optimization
        if hasattr(self, 'optimizer'):
            self.optimizer.step(loss)
        else:
            loss.backward()
            torch.optim.Adam(self.model.parameters()).step()
            torch.optim.Adam(self.model.parameters()).zero_grad()
        
        # Record metrics
        self.train_losses.append(loss.item())
        
        if hasattr(self, 'optimizer'):
            lr_info = self.optimizer.get_lr_info()
            self.lr_history.append(lr_info)
        
        return {
            "loss": loss.item(),
            "forward_time": forward_time,
            "predictions_shape": predictions.shape,
        }
    
    def validate(self) -> Dict[str, float]:
        """Run validation on validation set."""
        if self.val_dataloader is None:
            return {}
            
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                if "images" in batch and "token_ids" in batch:
                    predictions = self.model(
                        batch["images"], 
                        batch["token_ids"],
                        batch.get("attention_mask", None)
                    )
                else:
                    predictions = self.model(batch["observations"])
                
                # Compute loss
                targets = batch.get("actions", batch.get("targets"))
                loss = self.loss_fn(predictions, targets)
                val_losses.append(loss.item())
        
        val_loss = sum(val_losses) / len(val_losses)
        self.val_losses.append(val_loss)
        
        return {"val_loss": val_loss}
    
    def log_step(self, step_metrics: Dict[str, Any]) -> None:
        """Log training metrics."""
        print(f"Step {self.step:6d} | Loss: {step_metrics['loss']:.4f} | "
              f"Forward: {step_metrics['forward_time']*1000:.1f}ms")
        
        # Log quantization analysis periodically
        if self.step % (self.log_freq * 5) == 0:
            analyze_quantization(self.model, self.step, self.train_losses)
    
    def save_checkpoint(self, additional_info: Optional[Dict] = None) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "lr_history": self.lr_history,
        }
        
        if hasattr(self, 'optimizer'):
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save latest checkpoint
        latest_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint.get("val_losses", [])
        self.lr_history = checkpoint.get("lr_history", [])
        
        if hasattr(self, 'optimizer') and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"📁 Checkpoint loaded from: {checkpoint_path}")
    
    def train(
        self,
        num_steps: int = 10000,
        num_epochs: Optional[int] = None,
        resume_from: Optional[str] = None
    ) -> None:
        """
        Main training loop.
        
        Args:
            num_steps: Number of training steps (if num_epochs not specified)
            num_epochs: Number of epochs to train (overrides num_steps if specified)
            resume_from: Path to checkpoint to resume from
        """
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Setup defaults if not already configured
        if not hasattr(self, 'optimizer'):
            self.setup_optimizer()
        if not hasattr(self, 'loss_fn'):
            self.setup_loss_function()
        
        print(f"🚀 Starting training from step {self.step}")
        print(f"Target: {num_steps} steps" + (f" ({num_epochs} epochs)" if num_epochs else ""))
        
        start_time = time.time()
        
        # Training loop
        if num_epochs:
            # Epoch-based training
            for epoch in range(self.epoch, num_epochs):
                self.epoch = epoch
                for batch in self.train_dataloader:
                    step_metrics = self.train_step(batch)
                    self.step += 1
                    
                    # Logging
                    if self.step % self.log_freq == 0:
                        self.log_step(step_metrics)
                    
                    # Validation
                    if self.step % self.eval_freq == 0 and self.val_dataloader:
                        val_metrics = self.validate()
                        if val_metrics:
                            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                    
                    # Checkpointing
                    if self.step % self.save_freq == 0:
                        self.save_checkpoint()
                
                print(f"Epoch {epoch} completed")
        else:
            # Step-based training
            epoch_iter = iter(self.train_dataloader)
            
            while self.step < num_steps:
                try:
                    batch = next(epoch_iter)
                except StopIteration:
                    # New epoch
                    self.epoch += 1
                    epoch_iter = iter(self.train_dataloader)
                    batch = next(epoch_iter)
                
                step_metrics = self.train_step(batch)
                self.step += 1
                
                # Logging
                if self.step % self.log_freq == 0:
                    self.log_step(step_metrics)
                
                # Validation
                if self.step % self.eval_freq == 0 and self.val_dataloader:
                    val_metrics = self.validate()
                    if val_metrics:
                        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                
                # Checkpointing
                if self.step % self.save_freq == 0:
                    self.save_checkpoint()
        
        training_time = time.time() - start_time
        
        # Final checkpoint and summary
        self.save_checkpoint({"training_completed": True})
        
        print(f"\n✅ Training completed!")
        print(f"  Total time: {training_time:.1f} seconds")
        print(f"  Total steps: {self.step}")
        print(f"  Final loss: {self.train_losses[-1]:.4f}")
        print(f"  Output directory: {self.output_dir}")
        
        # Final quantization analysis
        if self.train_losses:
            print("\n🔍 Final Quantization Analysis:")
            analyze_quantization(self.model, self.step, self.train_losses) 