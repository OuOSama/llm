# src/train.py

import math
import os
import torch

from model import LLM, LLMConfig, device
from checkpoint import save_checkpoint, load_checkpoint


def get_lr(
    step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float
) -> float:
    """Linear warmup for `warmup_steps`, then cosine decay down to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


def train(
    model: LLM,
    get_batch,  # callable -> (x, y) LongTensors of shape (B, T), on CPU is fine
    max_steps: int = 10_000,
    warmup_steps: int = 200,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    grad_clip: float = 1.0,
    log_every: int = 100,
    checkpoint_dir: str = "checkpoints",
    save_every: int = 500,
    resume_from: str | None = None,
):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    on_cuda = device.type == "cuda"
    use_bf16 = on_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    # GradScaler only matters for fp16 (limited dynamic range).
    # bf16 doesn't need it, so we disable the scaler in that case.
    scaler = torch.amp.GradScaler("cuda", enabled=on_cuda and not use_bf16)

    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step, _ = load_checkpoint(resume_from, model, optimizer, scaler)
        print(f"resumed from {resume_from} at step {start_step}")

    model.train()
    for step in range(start_step, max_steps):
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=on_cuda):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % log_every == 0:
            print(f"step {step:6d} | lr {lr:.2e} | loss {loss.item():.4f}")

        if step > 0 and step % save_every == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, "latest.pt"),
                model,
                optimizer,
                step,
                model.config,
                scaler,
            )

    save_checkpoint(
        os.path.join(checkpoint_dir, "final.pt"),
        model,
        optimizer,
        max_steps,
        model.config,
        scaler,
    )
    return model


if __name__ == "__main__":
    # Quick architecture smoke test — NOT real training.
    # Real training entry point is src/main.py (uses real tokenized data).
    # This just checks the forward/backward/optimizer plumbing doesn't crash
    # and that loss actually goes down on a tiny random-data overfit.
    def dummy_get_batch(batch_size=8, block_size=128, vocab_size=1000):
        x = torch.randint(0, vocab_size, (batch_size, block_size))
        y = torch.randint(0, vocab_size, (batch_size, block_size))
        return x, y

    config = LLMConfig(vocab_size=1000, block_size=128, n_layer=4, n_head=4, n_embd=256)
    model = LLM(config)
    trained_model = train(model, dummy_get_batch, max_steps=500, warmup_steps=20)

    x, y = dummy_get_batch()
    _, final_loss = trained_model(x.to(device), y.to(device))
    print(f"final smoke-test loss: {final_loss.item():.4f}")
    assert final_loss.item() < 5.0, "loss didn't drop — check model/train plumbing!"
