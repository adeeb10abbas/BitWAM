"""Language encoder for the canonical 1-bit-ready VLA model."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear


@dataclass
class LanguageEncoderConfig:
    vocab_size: int = 8192
    embed_dim: int = 256
    hidden_dim: int = 512
    max_seq_len: int = 64
    dropout: float = 0.1


class LanguageEncoder(nn.Module):
    """Compact self-attention language encoder with BitLinear projections."""

    def __init__(
        self,
        vocab_size: int = 8192,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = LanguageEncoderConfig(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.02)
        self.attention_qkv = BitLinear(embed_dim, embed_dim * 3)
        self.attention_out = BitLinear(embed_dim, embed_dim)
        self.ffn1 = BitLinear(embed_dim, hidden_dim)
        self.ffn2 = BitLinear(hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = BitLinear(embed_dim, hidden_dim)

    def attention_forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seq_len, embed_dim = x.shape
        qkv = self.attention_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        scores = torch.bmm(q, k.transpose(1, 2)) / (embed_dim**0.5)
        if mask is not None:
            # mask True indicates valid token.
            key_mask = mask.unsqueeze(1).expand(bsz, seq_len, seq_len)
            scores = scores.masked_fill(~key_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)
        return self.attention_out(out)

    def forward(
        self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, seq_len = token_ids.shape
        x = self.embedding(token_ids)
        pos = self.pos_embedding[:seq_len].unsqueeze(0)
        x = x + pos
        attn_out = self.dropout(self.attention_forward(x, attention_mask))
        x = self.norm1(x + attn_out)
        ffn_out = self.dropout(self.ffn2(F.gelu(self.ffn1(x))))
        x = self.norm2(x + ffn_out)

        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = attention_mask.float().unsqueeze(-1)
            denom = torch.clamp(weights.sum(dim=1), min=1.0)
            pooled = (x * weights).sum(dim=1) / denom
        return self.output_proj(pooled)