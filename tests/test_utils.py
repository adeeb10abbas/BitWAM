"""Tests for utility modules."""

import torch

from bit_vla import VLABitNet
from bit_vla.utils import (
    SimpleTokenizer,
    absmax_quantize_activations,
    absmean_quantize_weights,
    build_task_prompt,
    create_sample_data,
)


def test_quantization_utils_shapes():
    w = torch.randn(8, 8)
    q_w, scale = absmean_quantize_weights(w)
    assert q_w.shape == w.shape
    assert scale.numel() == 1

    x = torch.randn(4, 8)
    q_x = absmax_quantize_activations(x)
    assert q_x.shape == x.shape


def test_tokenizer_batch_encode():
    tokenizer = SimpleTokenizer(vocab_size=1024)
    encoded = tokenizer.batch_encode(["move to target", "insert peg"], max_length=8)
    assert encoded["token_ids"].shape == (2, 8)
    assert encoded["attention_mask"].shape == (2, 8)


def test_task_prompts_non_empty():
    assert build_task_prompt("lerobot/pusht")
    assert build_task_prompt("lerobot/aloha_sim_insertion_human")


def test_sample_data_and_model_compatibility():
    data = create_sample_data(batch_size=2, max_seq_len=8, action_dim=2)
    model = VLABitNet(hidden_dim=64, action_dim=2, max_seq_len=8)
    with torch.no_grad():
        actions = model(data["images"], data["token_ids"], data["attention_mask"])
    assert actions.shape == (2, 2)
