# YAE-LLM 

A lightweight, educational implementation of a Transformer-based Large Language Model built with PyTorch. This project demonstrates the core components of modern LLM architecture with clear, readable code.

## 📁 Project Structure

```
src/
├── model/
│   ├── __init__.py
│   ├── Block.py          # Transformer block with pre-layer norm
│   ├── CausalSelfAttention.py  # Multi-head attention with RoPE and FlashAttention
│   ├── MLP.py            # SwiGLU feed-forward network
│   ├── model.py          # Main LLM class with config and generation
│   ├── RoPE.py           # Rotary Position Embedding utilities
│   └── HF_Warper.py      # Hugging Face compatibility wrapper
```

## 🧠 Architecture Highlights

### Rotary Position Embedding (RoPE)
- Implements relative position encoding via rotation matrices
- Supports **Long Context Scaling** with configurable `scaling_factor`
- Pre-computed caches for efficient inference

### Causal Self-Attention
- Multi-head attention with fused QKV projection
- Uses PyTorch's `scaled_dot_product_attention` with FlashAttention backend
- Integrated RoPE for position-aware attention

### SwiGLU MLP
- Modern feed-forward design with gated linear units
- Uses SiLU (Swish) activation
- Efficient hidden dimension sizing: `2 * (4 * n_embd) / 3`

### Pre-Layer Norm Architecture
- RMSNorm applied before each sub-layer
- Residual connections with dropout regularization
- Clean, modular block design

## ⚙️ Configuration

```python
@dataclass
class LLMConfig:
    vocab_size: int       # Vocabulary size
    block_size: int       # Maximum context length
    n_layer: int = 12     # Number of transformer blocks
    n_head: int = 12      # Number of attention heads
    n_embd: int = 768     # Embedding dimension
    dropout: float = 0.1
    rope_theta: float = 10000.0  # RoPE base frequency
```

## 🚀 Key Features

- **Weight Tying**: Embedding and output projection share weights
- **FlashAttention**: Accelerated attention computation via PyTorch's native kernel
- **RoPE Scaling**: Support for extended context windows via linear interpolation
- **Autoregressive Generation**: Temperature and top-k sampling for text generation
- **Hugging Face Compatible**: Easy integration with existing HF pipelines

## 💡 Usage Example

```python
from model.model import LLM, LLMConfig

# Configure the model
config = LLMConfig(
    vocab_size=32000,
    block_size=2048,
    n_layer=12,
    n_head=12,
    n_embd=768
)

# Initialize model
model = LLM(config)

# Forward pass with input tokens
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
logits, loss = model(input_ids, targets=target_ids)

# Generate text
output = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
```

## 🔧 Technical Notes

- **RoPE Implementation**: Uses `scaling_factor` for long context support (Qwen2/Llama3 style)
- **Memory Efficient**: Pre-computed RoPE caches registered as buffers (not trainable)
- **Flexible Head Dimension**: Supports arbitrary head sizes (must be even for RoPE)

## 📊 Dependencies

- PyTorch ≥ 2.0 (for `scaled_dot_product_attention`)
- Python ≥ 3.8
- (Optional) `transformers` for HF compatibility

## 🎯 Design Philosophy

This codebase prioritizes **readability** and **educational value** while maintaining production-quality architecture choices:

1. **Explicit Naming**: Chinese/English bilingual comments explain each component's purpose
2. **Modular Design**: Each architectural component lives in its own file
3. **Modern Practices**: Implements SOTA techniques (FlashAttention, SwiGLU, RoPE)
4. **Lean Implementation**: Minimal external dependencies, pure PyTorch

## 📝 License

MIT License - Feel free to use, modify, and learn from this code!