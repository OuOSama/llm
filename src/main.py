# src/main.py
from model import LLM, LLMConfig
from data import get_batch_fn
from train import train

VOCAB_SIZE = 50257  # ต้อง match กับ tiktoken "gpt2" encoding ที่ใช้ตอน prepare_data


def main():
    block_size = 256
    batch_size = 32

    config = LLMConfig(
        vocab_size=VOCAB_SIZE,
        block_size=block_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
    )
    model = LLM(config)

    get_batch = get_batch_fn(
        data_dir="data", block_size=block_size, batch_size=batch_size, split="train"
    )

    train(model, get_batch, max_steps=5000, warmup_steps=100)


if __name__ == "__main__":
    main()