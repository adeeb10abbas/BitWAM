"""
Standalone BitACT Policy for 1-bit VLA Research.

This is a simplified, standalone implementation of BitACT that can be used
independently of the LeRobot framework for research purposes.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from ..models.bitlinear import BitLinear


@dataclass
class BitACTConfig:
    """Configuration for BitACT policy."""
    
    # Model architecture
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 2048
    n_encoder_layers: int = 4
    n_decoder_layers: int = 7
    chunk_size: int = 50
    
    # BitNet quantization
    use_bitnet: bool = True
    bitnet_eps: float = 1e-5
    keep_fp16_layers: list = field(
        default_factory=lambda: ["action_head", "mu_head", "log_sigma_x2_head"]
    )
    
    # Training
    training_mode: str = "native_1bit"
    bitnet_lr_schedule: dict = field(default_factory=lambda: {
        "stage1_steps": 6000,
        "stage1_lr": 1e-3,
        "stage2_lr": 1e-4,
        "warmup_steps": 500,
    })
    
    # VAE settings
    use_vae: bool = True
    latent_dim: int = 32
    kl_weight: float = 10.0
    
    # Action settings
    n_action_steps: int = 8
    action_dim: int = 7
    
    def validate_bitnet_config(self):
        """Validate BitNet-specific configuration parameters."""
        if self.use_bitnet:
            assert self.bitnet_eps > 0, "BitNet epsilon must be positive"
            assert self.training_mode in [
                "native_1bit", "pretrain_finetune"
            ], f"Invalid training mode: {self.training_mode}"
            
            lr_schedule = self.bitnet_lr_schedule
            assert lr_schedule["stage1_steps"] > 0, (
                "Stage 1 steps must be positive"
            )
            assert lr_schedule["stage1_lr"] > lr_schedule["stage2_lr"], (
                "Stage 1 LR should be higher than stage 2"
            )


class SimplifiedTransformerLayer(nn.Module): 
    def __init__(self, config: BitACTConfig):
        super().__init__()
        self.config = config
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, batch_first=True
        )
        
        # Feed-forward network
        if config.use_bitnet:
            self.linear1 = BitLinear(config.dim_model, config.dim_feedforward)
            self.linear2 = BitLinear(config.dim_feedforward, config.dim_model)
        else:
            self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
            self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer layer."""
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.linear2(torch.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class BitACTPolicy(nn.Module):
    """
    Simplified BitACT Policy for research and education.
    
    This is a standalone implementation that demonstrates the core concepts
    of BitNet quantization applied to action chunking transformers.
    
    Args:
        config: BitACT configuration
        observation_dim: Dimension of observation space
        
    Example:
        >>> config = BitACTConfig(action_dim=7)
        >>> policy = BitACTPolicy(config, observation_dim=8)
        >>> obs = torch.randn(4, 8)
        >>> actions = policy(obs)
        >>> print(actions.shape)  # torch.Size([4, 8, 7])
    """
    
    def __init__(self, config: BitACTConfig, observation_dim: int = 8):
        super().__init__()
        self.config = config
        
        # Input projection
        self.obs_projection = nn.Linear(observation_dim, config.dim_model)
        
        # Positional encoding for action chunks
        self.pos_encoding = nn.Parameter(
            torch.randn(config.chunk_size, config.dim_model) * 0.1
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            SimplifiedTransformerLayer(config) 
            for _ in range(config.n_encoder_layers)
        ])
        
        # Transformer decoder layers  
        self.decoder_layers = nn.ModuleList([
            SimplifiedTransformerLayer(config)
            for _ in range(config.n_decoder_layers)
        ])
        
        # Action prediction head
        if "action_head" in config.keep_fp16_layers:
            self.action_head = nn.Linear(
                config.dim_model, config.action_dim
            )
        else:
            self.action_head = BitLinear(
                config.dim_model, config.action_dim
            )
        
        # VAE components (optional)
        if config.use_vae:
            if "mu_head" in config.keep_fp16_layers:
                self.mu_head = nn.Linear(config.dim_model, config.latent_dim)
                self.log_sigma_x2_head = nn.Linear(
                    config.dim_model, config.latent_dim
                )
            else:
                self.mu_head = BitLinear(config.dim_model, config.latent_dim)
                self.log_sigma_x2_head = BitLinear(
                    config.dim_model, config.latent_dim
                )
            
            self.latent_projection = nn.Linear(
                config.latent_dim, config.dim_model
            )
        
    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations through transformer encoder."""
        # Project observations to model dimension
        x = self.obs_projection(observations)  # [batch, dim_model]
        x = x.unsqueeze(1)  # [batch, 1, dim_model]
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x.squeeze(1)  # [batch, dim_model]
    
    def decode_actions(
        self, 
        encoded_obs: torch.Tensor, 
        latent: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode actions from encoded observations."""
        # Prepare decoder input
        if latent is not None:
            # Include latent information
            latent_features = self.latent_projection(latent)
            decoder_input = encoded_obs + latent_features
        else:
            decoder_input = encoded_obs
        
        # Expand for chunk_size and add positional encoding
        decoder_input = decoder_input.unsqueeze(1).expand(
            -1, self.config.chunk_size, -1
        )  # [batch, chunk_size, dim_model]
        decoder_input = decoder_input + self.pos_encoding.unsqueeze(0)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input)
        
        # Predict actions
        actions = self.action_head(decoder_input)
        
        return actions  # [batch, chunk_size, action_dim]
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BitACT policy.
        
        Args:
            observations: Input observations [batch_size, observation_dim]
            
        Returns:
            Action predictions [batch_size, chunk_size, action_dim]
        """
        # Encode observations
        encoded_obs = self.encode_observations(observations)
        
        # VAE encoding (if enabled)
        latent = None
        if self.config.use_vae:
            mu = self.mu_head(encoded_obs)
            log_sigma_x2 = self.log_sigma_x2_head(encoded_obs)
            
            if self.training:
                # Sample from latent distribution during training
                std = torch.exp(0.5 * log_sigma_x2)
                eps = torch.randn_like(std)
                latent = mu + eps * std
            else:
                # Use mean during inference
                latent = mu
        
        # Decode to actions
        actions = self.decode_actions(encoded_obs, latent)
        
        return actions
    
    def get_optim_params(self) -> list[dict]:
        """Get parameter groups for BitNet-specific optimization."""
        if not self.config.use_bitnet:
            return [{"params": self.parameters()}]
        
        bitnet_params = []
        fp16_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            # Check if this is a BitLinear parameter
            if any(layer_name in name for layer_name in 
                   ["linear1", "linear2"] if "action_head" not in name and 
                   "mu_head" not in name and "log_sigma_x2_head" not in name):
                bitnet_params.append(param)
            else:
                fp16_params.append(param)
        
        param_groups = [
            {"params": bitnet_params, "name": "bitnet_layers"},
            {"params": fp16_params, "name": "fp16_layers"},
        ]
        
        return param_groups
    
    def get_quantization_summary(self) -> dict:
        """Get quantization statistics for the model."""
        total_params = sum(p.numel() for p in self.parameters())
        
        bitlinear_params = 0
        bitlinear_layers = 0
        
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                bitlinear_layers += 1
                bitlinear_params += sum(
                    p.numel() for p in module.parameters()
                )
        
        fp16_params = total_params - bitlinear_params
        
        return {
            "total_parameters": total_params,
            "bitlinear_parameters": bitlinear_params,
            "fp16_parameters": fp16_params,
            "bitlinear_layers": bitlinear_layers,
            "quantized_ratio": bitlinear_params / total_params * 100,
            "estimated_size_mb": {
                "fp32": total_params * 4 / 1024**2,
                "quantized": (
                    bitlinear_params * 0.2 + fp16_params * 4
                ) / 1024**2,
            }
        } 