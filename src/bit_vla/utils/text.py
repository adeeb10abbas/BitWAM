"""Minimal deterministic text utilities for simulation training."""

from typing import Dict, List, Tuple

import torch


class SimpleTokenizer:
    """
    Lightweight tokenizer for deterministic text-conditioning without external deps.
    """

    def __init__(self, vocab_size: int = 8192, pad_id: int = 0, unk_id: int = 1):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.unk_id = unk_id

    def _token_to_id(self, token: str) -> int:
        if not token:
            return self.unk_id
        # Stable hash under Python process randomization.
        h = 2166136261
        for ch in token.lower():
            h = (h ^ ord(ch)) * 16777619
            h &= 0xFFFFFFFF
        return 2 + (h % max(1, self.vocab_size - 2))

    def encode(self, text: str, max_length: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = text.strip().split()
        token_ids = [self._token_to_id(tok) for tok in tokens[:max_length]]
        mask = [1] * len(token_ids)
        while len(token_ids) < max_length:
            token_ids.append(self.pad_id)
            mask.append(0)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def batch_encode(self, texts: List[str], max_length: int = 32) -> Dict[str, torch.Tensor]:
        ids, masks = zip(*(self.encode(t, max_length=max_length) for t in texts))
        return {
            "token_ids": torch.stack(list(ids), dim=0),
            "attention_mask": torch.stack(list(masks), dim=0),
        }


def build_task_prompt(dataset_name: str) -> str:
    name = dataset_name.lower()
    if "pusht" in name:
        return (
            "Push the T-shaped object to the target area while keeping smooth "
            "and stable end-effector motion."
        )
    if "aloha" in name:
        return (
            "Perform precise bimanual insertion with smooth coordinated arm "
            "control and avoid abrupt joint motion."
        )
    return "Follow the task objective with safe, smooth, and accurate control."
