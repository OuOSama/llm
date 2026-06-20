# src/main.py

from model import LLM, LLMConfig
from data import get_batch_fn
from train import train

VOCAB_SIZE = 50257


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
    get_val_batch = get_batch_fn(
        data_dir="data", block_size=block_size, batch_size=batch_size, split="val"
    )

    train(
        model,
        get_batch,
        get_val_batch=get_val_batch,
        max_steps=5000,
        warmup_steps=100,
        eval_interval=200,
        eval_iters=50,
        log_dir="runs/baseline",  # NEW — เปลี่ยนชื่อทุกครั้งที่ลอง config ใหม่ จะได้เทียบ curve กันใน dashboard ได้
    )


if __name__ == "__main__":
    main()
