"""
BitNet Optimizer with two-stage learning rate schedule.

This module implements optimization strategies specifically designed for BitNet models,
including the two-stage learning rate schedule that works well with quantized weights.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union


class BitNetOptimizer:
    """
    BitNet-specific optimizer wrapper that implements the two-stage learning rate schedule.
    
    This optimizer separates BitLinear layers and FP16 layers into different parameter 
    groups and applies a two-stage learning rate schedule optimized for quantized training.
    
    Args:
        model: Model to optimize (should contain BitLinear layers)
        stage1_lr: High learning rate for stage 1
        stage2_lr: Low learning rate for stage 2
        stage1_steps: Number of steps for stage 1
        weight_decay: Weight decay factor
        fp16_lr_ratio: LR ratio for FP16 layers (relative to BitLinear layers)
        
    Example:
        >>> model = VLABitNet(hidden_dim=512, action_dim=7)
        >>> optimizer = BitNetOptimizer(model, stage1_lr=1e-3, stage2_lr=1e-4, stage1_steps=5000)
        >>> 
        >>> for step, batch in enumerate(dataloader):
        >>>     optimizer.step_lr_schedule(step)  # Update LR
        >>>     loss = model(batch)
        >>>     optimizer.step(loss)
    """
    
    def __init__(
        self,
        model: nn.Module,
        stage1_lr: float = 1e-3,
        stage2_lr: float = 1e-4,
        stage1_steps: int = 6000,
        weight_decay: float = 0.1,
        fp16_lr_ratio: float = 0.5,
        warmup_steps: int = 500
    ):
        self.model = model
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        self.stage1_steps = stage1_steps
        self.weight_decay = weight_decay
        self.fp16_lr_ratio = fp16_lr_ratio
        self.warmup_steps = warmup_steps
        
        # Create parameter groups
        self.param_groups = self._create_param_groups()
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(self.param_groups, weight_decay=weight_decay)
        
        # Track current stage
        self.current_stage = 1
        self.current_step = 0
        
    def _create_param_groups(self) -> List[Dict]:
        """Create parameter groups for BitNet optimization."""
        bitnet_params = []
        fp16_params = []
        backbone_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Categorize parameters
            if self._is_bitlinear_param(name):
                bitnet_params.append(param)
            elif self._is_backbone_param(name):
                backbone_params.append(param) 
            else:
                fp16_params.append(param)
        
        param_groups = []
        
        if bitnet_params:
            param_groups.append({
                "params": bitnet_params,
                "name": "bitnet_layers",
                "lr": self.stage1_lr,
            })
            
        if fp16_params:
            param_groups.append({
                "params": fp16_params,
                "name": "fp16_layers", 
                "lr": self.stage1_lr * self.fp16_lr_ratio,
            })
            
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "name": "backbone",
                "lr": self.stage1_lr * 0.1,  # Lower LR for backbone
            })
        
        return param_groups
    
    def _is_bitlinear_param(self, param_name: str) -> bool:
        """Check if parameter belongs to a BitLinear layer."""
        bitlinear_indicators = [
            "linear1", "linear2", "projection", "feature_proj", "output_proj",
            "fusion_layer", "hidden_layer"
        ]
        # Exclude action heads which should stay FP16
        exclude_indicators = ["action_head", "mu_head", "log_sigma_x2_head"]
        
        has_bitlinear = any(indicator in param_name for indicator in bitlinear_indicators)
        has_exclude = any(indicator in param_name for indicator in exclude_indicators)
        
        return has_bitlinear and not has_exclude
    
    def _is_backbone_param(self, param_name: str) -> bool:
        """Check if parameter belongs to backbone (conv layers, embeddings)."""
        backbone_indicators = ["conv", "embedding", "pos_encoding"]
        return any(indicator in param_name for indicator in backbone_indicators)
    
    def step_lr_schedule(self, step: int) -> None:
        """Update learning rate according to two-stage schedule."""
        self.current_step = step
        
        # Check for stage transition
        if step == self.stage1_steps and self.current_stage == 1:
            self._transition_to_stage2()
        
        # Apply warmup if in early training
        if step < self.warmup_steps:
            self._apply_warmup(step)
    
    def _transition_to_stage2(self) -> None:
        """Transition from stage 1 to stage 2."""
        print(f"🔄 BitNet Optimizer: Transitioning to Stage 2 at step {self.current_step}")
        self.current_stage = 2
        
        for group in self.optimizer.param_groups:
            if group.get("name") == "bitnet_layers":
                group["lr"] = self.stage2_lr
            elif group.get("name") == "fp16_layers":
                group["lr"] = self.stage2_lr * self.fp16_lr_ratio
            elif group.get("name") == "backbone":
                group["lr"] = self.stage2_lr * 0.1
            
            # Set weight decay to zero in stage 2
            group["weight_decay"] = 0.0
    
    def _apply_warmup(self, step: int) -> None:
        """Apply learning rate warmup."""
        warmup_factor = min(1.0, step / self.warmup_steps)
        
        for group in self.optimizer.param_groups:
            base_lr = self.stage1_lr if self.current_stage == 1 else self.stage2_lr
            
            if group.get("name") == "bitnet_layers":
                group["lr"] = base_lr * warmup_factor
            elif group.get("name") == "fp16_layers":
                group["lr"] = base_lr * self.fp16_lr_ratio * warmup_factor
            elif group.get("name") == "backbone":
                group["lr"] = base_lr * 0.1 * warmup_factor
    
    def step(self, loss: torch.Tensor) -> None:
        """Perform optimization step."""
        loss.backward()
        
        # Optional gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self) -> Dict:
        """Get optimizer state dict."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "current_stage": self.current_stage,
            "current_step": self.current_step,
            "stage1_lr": self.stage1_lr,
            "stage2_lr": self.stage2_lr,
            "stage1_steps": self.stage1_steps,
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.current_stage = state_dict["current_stage"]
        self.current_step = state_dict["current_step"]
        self.stage1_lr = state_dict["stage1_lr"]
        self.stage2_lr = state_dict["stage2_lr"]
        self.stage1_steps = state_dict["stage1_steps"]
    
    def get_lr_info(self) -> Dict:
        """Get current learning rate information."""
        lr_info = {
            "current_stage": self.current_stage,
            "current_step": self.current_step,
            "stage1_steps": self.stage1_steps,
        }
        
        for group in self.optimizer.param_groups:
            group_name = group.get("name", "unknown")
            lr_info[f"{group_name}_lr"] = group["lr"]
        
        return lr_info 