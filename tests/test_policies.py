"""Tests for BitACT policy API sanity."""

import torch

from bit_vla.policies import BitACTConfig, BitACTPolicy


def test_bitact_config_fields():
    cfg = BitACTConfig(action_dim=7, chunk_size=8, dim_model=128)
    assert cfg.action_dim == 7
    assert cfg.chunk_size == 8
    assert cfg.dim_model == 128


def test_bitact_forward_shape():
    cfg = BitACTConfig(action_dim=7, chunk_size=8, dim_model=128, use_vae=False)
    policy = BitACTPolicy(cfg, observation_dim=16)
    obs = torch.randn(2, 16)
    out = policy(obs)
    assert out.shape == (2, 8, 7)


def test_bitact_quantization_summary():
    cfg = BitACTConfig(action_dim=7, chunk_size=8, use_vae=False)
    policy = BitACTPolicy(cfg, observation_dim=8)
    summary = policy.get_quantization_summary()
    assert "total_parameters" in summary
    assert "quantized_ratio" in summary
