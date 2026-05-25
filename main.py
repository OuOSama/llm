import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # word token embedding (WTE)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # word positional embedding (WPE)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The single, fused linear layer for Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Get Q, K, V from a single projection and split them
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # 2. Calculate attention weights
        scaled_scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attention_weights = F.softmax(scaled_scores, dim=-1)

        # 3. Aggregate values
        output = attention_weights @ v
        return output


class CausalSelfAttention(nn.Module):
    bias: torch.Tensor

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Get QKV and SPLIT into heads
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)

        # 2. Run causal self-ATTENTION on each head in parallel
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, hd)

        # 3. MERGE heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
