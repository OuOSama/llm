import torch.nn as nn

from src.model.config import LLMConfig
from src.model.MLP import MLP
from src.model.CausalSelfAttention import CausalSelfAttention


class Block(nn.Module):
    """
    A single Transformer block consisting of an attention mechanism
    followed by a feed-forward MLP, organized with Pre-LayerNorm architecture.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        # RMSNorm for better training stability and performance
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        # Pre-LayerNorm: apply normalization before the attention layer,
        # then add the result to the original input (Residual Connection)
        x = x + self.attn(self.ln_1(x), cos, sin)

        # Pre-LayerNorm: apply normalization before the MLP layer,
        # then add the result to the current stream (Residual Connection)
        x = x + self.mlp(self.ln_2(x))
        return x
