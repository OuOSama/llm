import torch
import torch.nn as nn
import torch.nn.functional as F
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LLM(nn.Module):
    def __init__(self, config):
        # word token embedding (WTE)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # word positional embedding (WPE)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The single, fused linear layer for Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

    def forward(self, x):
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
