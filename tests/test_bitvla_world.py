import pytest
import torch

from lerobot_policy_bitwam.bitvla_world import LatentWorldModelHead, future_observation_indices


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
