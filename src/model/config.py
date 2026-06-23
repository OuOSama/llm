# src/model/config.py

from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration class for the LLM architecture."""

    vocab_size: int
    block_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    rope_theta: float = 10000.0
