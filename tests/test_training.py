"""Tests for training utilities."""

import torch

from bit_vla.models import VLABitNet
from bit_vla.training import BitNetOptimizer


def test_bitnet_optimizer_init_with_model():
    model = VLABitNet(hidden_dim=128, action_dim=2, state_dim=4, max_seq_len=16)
    optimizer = BitNetOptimizer(model)
    assert len(optimizer.optimizer.param_groups) >= 1


def test_bitnet_optimizer_step_flow():
    model = VLABitNet(hidden_dim=64, action_dim=2, state_dim=4, max_seq_len=8)
    optimizer = BitNetOptimizer(model, stage1_steps=2, warmup_steps=1)

    images = torch.randn(2, 3, 64, 64)
    token_ids = torch.randint(1, 200, (2, 8))
    states = torch.randn(2, 4)
    targets = torch.randn(2, 2)

    pred = model(images=images, token_ids=token_ids, states=states)
    loss = torch.nn.functional.mse_loss(pred, targets)
    optimizer.step_lr_schedule(0)
    optimizer.step(loss)

    lr_info = optimizer.get_lr_info()
    assert "current_stage" in lr_info


def test_optimizer_stage_transition():
    model = VLABitNet(hidden_dim=64, action_dim=2, state_dim=4, max_seq_len=8)
    optimizer = BitNetOptimizer(model, stage1_steps=1, warmup_steps=0)
    optimizer.step_lr_schedule(1)
    assert optimizer.current_stage == 2
