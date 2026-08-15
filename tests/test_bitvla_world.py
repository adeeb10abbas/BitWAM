import pytest
import torch

from lerobot_policy_bitwam.bitvla_world import (
    LatentWorldModelHead,
    TernaryLinear,
    future_observation_indices,
)


def test_future_observation_indices_follow_the_complete_action_chunk() -> None:
    assert future_observation_indices(12, 8) == (8, 9, 10, 11)
    assert future_observation_indices(8, 8) == ()


def test_future_observation_indices_validate_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        future_observation_indices(0, 0)
    with pytest.raises(ValueError, match="non-negative"):
        future_observation_indices(10, -1)


def test_latent_world_head_has_finite_loss_and_gradients() -> None:
    torch.manual_seed(0)
    head = LatentWorldModelHead(
        16,
        action_chunk_size=2,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=24,
    )
    hidden = torch.randn(4, 6, 16, requires_grad=True)
    actions = torch.randn(4, 2, 3)
    target = torch.randn(4, 16, requires_grad=True)

    output = head(hidden, actions, target)
    output.loss.backward()

    assert output.prediction.shape == (4, 16)
    assert torch.isfinite(output.loss)
    assert -1 <= output.cosine_similarity.item() <= 1
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) > 0
    assert target.grad is None
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_latent_world_head_rejects_wrong_action_shape() -> None:
    head = LatentWorldModelHead(16, action_chunk_size=2, action_dim=3)
    with pytest.raises(ValueError, match="actions must have shape"):
        head.predict(torch.randn(1, 6, 16), torch.randn(1, 3, 3))


def test_ternary_world_head_uses_ternary_matrices_and_straight_through_gradients() -> None:
    torch.manual_seed(0)
    head = LatentWorldModelHead(
        16,
        action_chunk_size=2,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=24,
        ternary=True,
    )
    ternary_layers = [module for module in head.modules() if isinstance(module, TernaryLinear)]
    assert len(ternary_layers) == 3
    for layer in ternary_layers:
        effective = layer.effective_weight().float()
        scale = layer.weight.detach().float().abs().mean().clamp(min=1e-5)
        levels = torch.unique((effective / scale).round())
        assert set(levels.tolist()) <= {-1.0, 0.0, 1.0}

    output = head(torch.randn(4, 5, 16), torch.randn(4, 2, 3), torch.randn(4, 16))
    output.loss.backward()
    assert all(layer.weight.grad is not None for layer in ternary_layers)
