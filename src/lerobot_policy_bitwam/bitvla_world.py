"""Future-latent world-model components for a native ternary BitVLA policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class _TernaryWeight(torch.autograd.Function):
    """Absmean ternary quantization with a straight-through gradient."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        del ctx
        dtype = weight.dtype
        weight_float = weight.float()
        scale = weight_float.abs().mean().clamp(min=1e-5)
        quantized = (weight_float / scale).round().clamp(-1, 1) * scale
        return quantized.to(dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        return (gradient,)


class _EightBitActivation(torch.autograd.Function):
    """Per-token absmax INT8 simulation with a straight-through gradient."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        del ctx
        dtype = inputs.dtype
        inputs_float = inputs.float()
        scale = 127 / inputs_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        quantized = (inputs_float * scale).round().clamp(-128, 127) / scale
        return quantized.to(dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        return (gradient,)


class TernaryLinear(nn.Linear):
    """BitNet-compatible 1.58-bit weights and per-token 8-bit activations."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantized_inputs = _EightBitActivation.apply(inputs)
        quantized_weight = _TernaryWeight.apply(self.weight)
        return F.linear(quantized_inputs, quantized_weight, self.bias)

    def effective_weight(self) -> torch.Tensor:
        """Return the dequantized ternary matrix used by the forward pass."""
        return _TernaryWeight.apply(self.weight.detach())


def future_observation_indices(trajectory_length: int, horizon: int) -> tuple[int, ...]:
    """Return target-frame indices after executing an action chunk of ``horizon`` steps."""
    if trajectory_length < 1:
        raise ValueError("trajectory_length must be positive")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if horizon >= trajectory_length:
        return ()
    return tuple(range(horizon, trajectory_length))


@dataclass(frozen=True)
class WorldModelOutput:
    """Prediction and normalized representation loss from the world head."""

    prediction: torch.Tensor
    loss: torch.Tensor
    cosine_similarity: torch.Tensor


class LatentWorldModelHead(nn.Module):
    """Predict a future VLA visual latent from policy state and an action chunk."""

    def __init__(
        self,
        latent_dim: int,
        *,
        action_chunk_size: int = 8,
        action_dim: int = 7,
        action_embedding_dim: int = 256,
        hidden_dim: int = 2048,
        ternary: bool = False,
    ) -> None:
        super().__init__()
        if min(latent_dim, action_chunk_size, action_dim, action_embedding_dim, hidden_dim) < 1:
            raise ValueError("world-model dimensions must be positive")
        self.latent_dim = latent_dim
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.ternary = ternary
        linear = TernaryLinear if ternary else nn.Linear
        self.state_norm = nn.LayerNorm(latent_dim)
        self.action_encoder = nn.Sequential(
            linear(action_chunk_size * action_dim, action_embedding_dim),
            nn.SiLU(),
        )
        self.predictor = nn.Sequential(
            linear(latent_dim + action_embedding_dim, hidden_dim),
            nn.SiLU(),
            linear(hidden_dim, latent_dim),
        )

    def predict(self, action_hidden_states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict the future visual representation without computing a loss."""
        if action_hidden_states.ndim != 3:
            raise ValueError("action_hidden_states must have shape [batch, tokens, latent_dim]")
        if action_hidden_states.shape[-1] != self.latent_dim:
            raise ValueError(
                f"expected action hidden size {self.latent_dim}, got {action_hidden_states.shape[-1]}"
            )
        expected_action_shape = (self.action_chunk_size, self.action_dim)
        if actions.ndim != 3 or tuple(actions.shape[1:]) != expected_action_shape:
            raise ValueError(
                "actions must have shape "
                f"[batch, {self.action_chunk_size}, {self.action_dim}], got {tuple(actions.shape)}"
            )
        if actions.shape[0] != action_hidden_states.shape[0]:
            raise ValueError("action hidden states and actions must have the same batch size")

        policy_state = self.state_norm(action_hidden_states.mean(dim=1))
        action_state = self.action_encoder(actions.flatten(start_dim=1))
        return self.predictor(torch.cat((policy_state, action_state), dim=-1))

    def forward(
        self,
        action_hidden_states: torch.Tensor,
        actions: torch.Tensor,
        future_visual_latent: torch.Tensor,
    ) -> WorldModelOutput:
        """Compute a scale-invariant future-representation prediction objective."""
        prediction = self.predict(action_hidden_states, actions)
        if future_visual_latent.shape != prediction.shape:
            raise ValueError(
                "future_visual_latent must match the prediction shape, "
                f"got {tuple(future_visual_latent.shape)} and {tuple(prediction.shape)}"
            )
        normalized_prediction = F.normalize(prediction.float(), dim=-1)
        normalized_target = F.normalize(future_visual_latent.detach().float(), dim=-1)
        cosine_similarity = (normalized_prediction * normalized_target).sum(dim=-1).mean()
        loss = 1.0 - cosine_similarity
        return WorldModelOutput(
            prediction=prediction,
            loss=loss,
            cosine_similarity=cosine_similarity,
        )
