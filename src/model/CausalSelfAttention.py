# src/model/CausalSelfAttention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import LLMConfig
from src.model.RoPE import apply_rope


class CausalSelfAttention(nn.Module):
    """
    Computes Multi-Head Causal Self-Attention with Rotary Positional Embeddings (RoPE).

    This module implements the core attention mechanism used in Transformer-based LLMs,
    incorporating a causal mask to ensure that each token can only attend to preceding tokens.
    It leverages FlashAttention for memory-efficient and high-performance computation.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        # Ensure embedding dimension is divisible by the number of heads
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Single linear layer to project input into Query, Key, and Value components simultaneously
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # Output projection layer to map attention results back to embedding dimension
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension

        # 1. Project input to Q, K, V and split into three separate tensors
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2. Reshape into multi-head format: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Apply Rotary Positional Embeddings (RoPE) to Q and K
        # Casting to x.dtype ensures numerical stability across different precision settings
        q = apply_rope(q, cos, sin).to(x.dtype)
        k = apply_rope(k, cos, sin).to(x.dtype)

        # 4. Perform Flash Attention (fused kernel) with causal masking
        # This is memory-efficient and automatically optimized for performance
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # 5. Concatenate heads back to original shape (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 6. Apply final linear projection and residual dropout
        y = self.resid_drop(self.c_proj(y))
        return y
