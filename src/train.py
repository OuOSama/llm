# src/train.py

import math
import os
import torch
from torch.utils.tensorboard import SummaryWriter

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
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model, batch_fns: dict, eval_iters: int, amp_dtype, on_cuda: bool
) -> dict:
    was_training = model.training
    model.eval()
    out = {}
    for split, get_batch in batch_fns.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=on_cuda
            ):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if was_training:
        model.train()
    return out


def train(
    model: LLM,
    get_batch,
    get_val_batch=None,
    max_steps: int = 10_000,
    warmup_steps: int = 200,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    grad_clip: float = 1.0,
    log_every: int = 100,
    eval_interval: int = 200,
    eval_iters: int = 50,
    checkpoint_dir: str = "checkpoints",
    save_every: int = 500,
    resume_from: str | None = None,
    auto_resume: bool = True,
    log_dir: str = "runs",  # NEW — TensorBoard event files go here
):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    on_cuda = device.type == "cuda"
    use_bf16 = on_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=on_cuda and not use_bf16)

    if resume_from is None and auto_resume:
        candidate = os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(candidate):
            resume_from = candidate
            print(f"found existing checkpoint at {candidate} — auto-resuming")

    start_step = 0
    best_val_loss = float("inf")
    if resume_from and os.path.exists(resume_from):
        start_step, _, best_val_loss = load_checkpoint(
            resume_from, model, optimizer, scaler
        )
        print(
            f"resumed from {resume_from} at step {start_step} "
            f"(best val loss so far: {best_val_loss:.4f})"
        )

    if start_step >= max_steps:
        print(
            f"checkpoint already at step {start_step} >= max_steps {max_steps}, nothing to do"
        )
        return model

    # NEW — เปิด writer แค่ตอนที่จะเทรนจริง (ไม่งั้นมันจะไปสร้าง log_dir เปล่า ๆ ทิ้งไว้)
    writer = SummaryWriter(log_dir=log_dir)

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

        # NEW — log ทุก step ขึ้น TensorBoard (ถูกมาก แค่เขียน scalar เล็ก ๆ)
        writer.add_scalar("lr", lr, step)
        writer.add_scalar("loss/train_step", loss.item(), step)

        if step % log_every == 0:
            print(f"step {step:6d} | lr {lr:.2e} | train loss {loss.item():.4f}")

        if get_val_batch is not None and step > 0 and step % eval_interval == 0:
            losses = estimate_loss(
                model,
                {"train": get_batch, "val": get_val_batch},
                eval_iters=eval_iters,
                amp_dtype=amp_dtype,
                on_cuda=on_cuda,
            )
            writer.add_scalar("loss/eval_train", losses["train"], step)
            writer.add_scalar("loss/eval_val", losses["val"], step)
            print(
                f"step {step:6d} | eval: train {losses['train']:.4f} | val {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    os.path.join(checkpoint_dir, "best.pt"),
                    model,
                    optimizer,
                    step,
                    model.config,
                    scaler,
                    best_val_loss=best_val_loss,
                )
                print("  -> new best val loss, saved best.pt")

        if step > 0 and step % save_every == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, "latest.pt"),
                model,
                optimizer,
                step,
                model.config,
                scaler,
                best_val_loss=best_val_loss,
            )

    save_checkpoint(
        os.path.join(checkpoint_dir, "final.pt"),
        model,
        optimizer,
        max_steps,
        model.config,
        scaler,
        best_val_loss=best_val_loss,
    )
    writer.close()  # NEW — flush ทุกอย่างลง disk ก่อนจบ
    return model


if __name__ == "__main__":

    def dummy_get_batch(batch_size=8, block_size=128, vocab_size=1000):
        x = torch.randint(0, vocab_size, (batch_size, block_size))
        y = torch.randint(0, vocab_size, (batch_size, block_size))
        return x, y

    config = LLMConfig(vocab_size=1000, block_size=128, n_layer=4, n_head=4, n_embd=256)
    model = LLM(config)
    trained_model = train(
        model,
        dummy_get_batch,
        max_steps=500,
        warmup_steps=20,
        auto_resume=False,
        log_dir="runs/smoke_test",
    )

    x, y = dummy_get_batch()
    _, final_loss = trained_model(x.to(device), y.to(device))
    print(f"final smoke-test loss: {final_loss.item():.4f}")
    assert final_loss.item() < 5.0, "loss didn't drop — check model/train plumbing!"
