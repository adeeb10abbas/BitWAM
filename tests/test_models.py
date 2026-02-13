"""Tests for canonical VLA model components."""

import torch

from bit_vla.models import ActionDecoder, BitLinear, LanguageEncoder, VLABitNet, VisionEncoder


def test_bitlinear_forward_and_summary():
    layer = BitLinear(32, 16)
    out = layer(torch.randn(4, 32))
    assert out.shape == (4, 16)
    summary = layer.get_quantization_summary()
    assert summary["total_parameters"] > 0
    assert summary["quantized_parameters"] > 0


def test_vision_encoder_shape():
    encoder = VisionEncoder(output_dim=128)
    features = encoder(torch.randn(2, 3, 96, 96))
    assert features.shape == (2, 128)


def test_language_encoder_shape_and_mask():
    encoder = LanguageEncoder(vocab_size=2048, hidden_dim=128, max_seq_len=16)
    token_ids = torch.randint(1, 100, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.bool)
    features = encoder(token_ids, mask)
    assert features.shape == (2, 128)


def test_action_decoder_shape():
    decoder = ActionDecoder(input_dim=128, hidden_dim=64, action_dim=7)
    actions = decoder(torch.randn(2, 128))
    assert actions.shape == (2, 7)


def test_vla_bitnet_forward_without_state():
    model = VLABitNet(hidden_dim=128, action_dim=7, state_dim=0, max_seq_len=16)
    images = torch.randn(2, 3, 96, 96)
    token_ids = torch.randint(1, 500, (2, 16))
    actions = model(images=images, token_ids=token_ids)
    assert actions.shape == (2, 7)


def test_vla_bitnet_forward_with_state():
    model = VLABitNet(hidden_dim=128, action_dim=2, state_dim=4, max_seq_len=16)
    batch = {
        "images": torch.randn(2, 3, 96, 96),
        "token_ids": torch.randint(1, 500, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.bool),
        "states": torch.randn(2, 4),
    }
    actions = model.forward_from_batch(batch)
    assert actions.shape == (2, 2)


def test_vla_quantization_summary_schema():
    model = VLABitNet(hidden_dim=128, action_dim=7)
    summary = model.get_quantization_summary()
    assert "total_parameters" in summary
    assert "quantized_parameters" in summary
    assert "quantized_ratio" in summary
    assert 0.0 <= summary["quantized_ratio"] <= 100.0
