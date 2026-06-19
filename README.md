# LLM Implementation with RoPE and Flash Attention

## Overview
This repository contains a clean, modern implementation of a decoder-only transformer language model featuring **Rotary Position Embeddings (RoPE)** and **Flash Attention** support. The implementation replaces traditional learned positional embeddings with RoPE and leverages PyTorch's `scaled_dot_product_attention` for efficient attention computation.

## Architecture Highlights

### Key Features
- **Rotary Position Embeddings (RoPE)**: Instead of learned positional embeddings, this model applies rotation matrices to query and key vectors based on their positions, providing better extrapolation capabilities.
- **Flash Attention**: Uses PyTorch's `F.scaled_dot_product_attention` which automatically selects optimal attention kernels (FlashAttention, memory-efficient attention) based on hardware.
- **Weight Tying**: The input embedding weights are tied with the output projection layer (`lm_head`) for parameter efficiency.
- **BF16/FP16 Support**: Automatic mixed precision training with adaptive scaling.

## Model Components

### 1. **LLMConfig**
Configuration dataclass with parameters:
- `vocab_size`: Vocabulary size
- `block_size`: Maximum sequence length
- `n_layer/n_head/n_embd`: Model architecture dimensions
- `dropout`: Dropout rate
- `rope_theta`: Base frequency for RoPE (default: 10000.0)

### 2. **RoPE Implementation**
```python
# Pre-computed sin/cos tables for all positions
cos, sin = build_rope_cache(head_dim, max_seq_len, theta)
# Applied to queries and keys
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
```

### 3. **CausalSelfAttention**
- Projects input to Q, K, V
- Applies RoPE to Q and K
- Uses `scaled_dot_product_attention` with causal masking
- Handles dropout only during training

### 4. **Training Pipeline**
- **Learning Rate Schedule**: Linear warmup followed by cosine decay
- **Optimizer**: AdamW with weight decay
- **Gradient Clipping**: Prevents gradient explosion
- **Mixed Precision**: Automatic with BF16/FP16 support

## Usage

### Training Example
```python
from model import LLM, LLMConfig
from train import train

config = LLMConfig(
    vocab_size=50000,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768
)

model = LLM(config)
train(model, your_data_loader, max_steps=100000)
```

### Text Generation
```python
model.eval()
input_ids = torch.tensor([[1, 2, 3]])  # Start tokens
generated = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
```

## Performance Optimizations

1. **RoPE Pre-computation**: Sin/cos tables computed once and reused
2. **Flash Attention**: Automatic kernel selection for optimal performance
3. **Mixed Precision**: BF16 on compatible GPUs, FP16 otherwise
4. **Memory Efficient**: No separate positional embeddings, tied weights

## Requirements
- PyTorch 2.0+ (for `scaled_dot_product_attention`)
- CUDA-capable GPU recommended

## Architecture Diagram
```
Input Tokens → Embedding → Dropout
    ↓
[Block × n_layer]
    ├── LayerNorm
    ├── Multi-Head Self-Attention (with RoPE)
    │   ├── QKV Projection
    │   ├── RoPE on Q/K
    │   └── Flash Attention
    ├── Residual Connection
    ├── LayerNorm
    └── MLP (GELU)
    ↓
Final LayerNorm → Linear (tied weights) → Logits
```

## Differences from Original Transformer
- **No learned positional embeddings**: Replaced with RoPE
- **No manual attention implementation**: Uses PyTorch's optimized `scaled_dot_product_attention`
- **Simpler code**: Cleaner, more maintainable implementation

## Notes
- The dummy data loader in `train.py` should be replaced with your actual dataset
- Configuration can be scaled to larger models (GPT-2, GPT-3 sizes) by adjusting config parameters
- The implementation is suitable for both research and production use

## License
[MIT License] - Feel free to use, modify, and distribute.