# src/model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.RoPE import build_rope_cache
from src.model.Block import Block
from src.model.config import LLMConfig


class LLM(nn.Module):
    """
    The main Transformer-based Language Model.

    This class handles the token embedding, stacked transformer blocks,
    RMSNorm, and the final language modeling head. It also manages
    precomputed RoPE caches for positional information.
    """

    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # Token embedding and dropout
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Precompute RoPE cache to be used across all attention blocks
        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope_cache(
            head_dim=head_dim, max_seq_len=config.block_size, theta=config.rope_theta
        )
        # Register as buffers so they are moved to the same device as the model
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

        # Stack of Transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between embedding and output projection
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.

        Args:
            idx: Input token indices (B, T)
            targets: Optional ground truth labels for loss calculation
        Returns:
            logits: Output predictions (B, T, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        B, T = idx.size()
        x = self.drop(self.wte(idx))

        # Retrieve the relevant portion of the RoPE cache
        cos_slot = self.cos_cached[:T].to(x.dtype)
        sin_slot = self.sin_cached[:T].to(x.dtype)

        # Pass through all transformer blocks
        for block in self.h:
            x = block(x, cos_slot, sin_slot)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Autoregressive text generation using multinomial sampling.

        Args:
            idx: Initial context indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Softmax scaling factor for output randomness
            top_k: Limit sampling to the top-k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds the model's block_size
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)

            # Apply temperature scaling
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < thresh, torch.full_like(logits, -float("inf")), logits
                )

            # Sample from the probability distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
