"""
Language encoder for 1-bit VLA models.

This module implements a language encoder that processes text instructions 
into feature representations using BitNet quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .bitlinear import BitLinear


class LanguageEncoder(nn.Module):
    """
    Simple language encoder using BitNet principles.
    
    Processes text instructions to feature representations using token 
    embeddings followed by BitLinear transformer-like layers.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Token embedding dimension
        hidden_dim: Hidden dimension for processing
        max_seq_len: Maximum sequence length for positional encoding
        
    Example:
        >>> encoder = LanguageEncoder(vocab_size=1000, hidden_dim=512)
        >>> tokens = torch.randint(0, 1000, (4, 32))
        >>> features = encoder(tokens)
        >>> print(features.shape)  # torch.Size([4, 512])
    """
    
    def __init__(
        self, 
        vocab_size: int = 10000, 
        embed_dim: int = 256, 
        hidden_dim: int = 512, 
        max_seq_len: int = 128
    ):
        super().__init__()
        
        # Token embedding (not quantized for simplicity)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_len, embed_dim) * 0.1
        )
        
        # BitNet transformer-like layers
        self.attention_qkv = BitLinear(embed_dim, embed_dim * 3)
        self.attention_out = BitLinear(embed_dim, embed_dim)
        self.ffn1 = BitLinear(embed_dim, hidden_dim)
        self.ffn2 = BitLinear(hidden_dim, embed_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.output_proj = BitLinear(embed_dim, hidden_dim)
        
    def attention_forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Simple self-attention with BitLinear layers.
        
        Args:
            x: Input embeddings [batch_size, seq_len, embed_dim]
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Attention output [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V using BitLinear
        qkv = self.attention_qkv(x)  # [batch, seq, embed_dim * 3]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, seq, embed_dim]
        
        # Simple dot-product attention (single head for simplicity)
        scores = torch.bmm(q, k.transpose(1, 2)) / (embed_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            # Convert boolean mask to attention mask
            mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.bmm(attn_weights, v)
        
        # Output projection
        output = self.attention_out(attn_output)
        
        return output
        
    def forward(
        self, 
        token_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through language encoder.
        
        Args:
            token_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Language features [batch_size, hidden_dim]
        """
        batch_size, seq_len = token_ids.shape
        
        # Token embedding + positional encoding
        x = self.embedding(token_ids)  # [batch, seq, embed_dim]
        
        # Add positional encoding (truncate or pad as needed)
        pos_enc = self.pos_encoding[:seq_len, :]
        x = x + pos_enc.unsqueeze(0)
        
        # Self-attention block with residual connection
        attn_out = self.attention_forward(x, attention_mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward block with residual connection
        ffn_out = self.ffn2(F.relu(self.ffn1(x)))
        x = self.norm2(x + ffn_out)
        
        # Global average pooling (considering mask)
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x_masked = x * mask_expanded
            sequence_features = x_masked.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple average pooling
            sequence_features = x.mean(dim=1)
        
        # Final output projection
        language_features = self.output_proj(sequence_features)
        
        return language_features 