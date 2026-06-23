# src/model/MLP.py

import torch.nn as nn
import torch.nn.functional as F

from src.model.config import LLMConfig


class MLP(nn.Module):
    """
    Multi-Layer Perceptron using SwiGLU activation.

    This architecture uses a gated linear unit with Swish (SiLU) activation,
    which is standard in modern LLMs (like Llama and Qwen) to improve
    representation power and training stability.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        # Calculate hidden dimension using standard SwiGLU scaling (approx 2/3 of 4x)
        hidden_dim = int(2 * (4 * config.n_embd) / 3)

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)  # Down-projection
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # Value projection
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU operation: (SiLU(x @ w1) * (x @ w3)) @ w2
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))
