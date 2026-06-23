# src/model/RoPE.py

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits the last dimension of the input into two halves,
    negates the second half, and swaps them to perform vector rotation.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Positional Embeddings (RoPE) to the input tensor.

    Uses the formula: x_rotated = (x * cos) + (rotate_half(x) * sin).
    Broadcasts cos and sin tensors to match the input shape (B, n_head, T, head_dim).
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (rotate_half(x) * sin)


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    device=None,
):
    """
    Precomputes the cos and sin cache for Rotary Positional Embeddings.

    Supports Linear Interpolation scaling to allow for longer context windows
    than the original model was trained on.
    """
    assert head_dim % 2 == 0, "head_dim must be divisible by 2"

    # Compute inverse frequencies
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    # Apply scaling factor for long-context support
    inv_freq = inv_freq / scaling_factor

    # Compute rotation frequencies for each position
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)

    # Duplicate frequencies to match head_dim and return cos/sin values
    emb = torch.cat((freqs, freqs), dim=-1)

    return emb.cos(), emb.sin()
