# src/checkpoint.py

import os
import dataclasses
import torch


def save_checkpoint(path, model, optimizer, step, config, scaler=None):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": dataclasses.asdict(config),  # plain dict — ปลอดภัยกว่าเก็บ object ตรงๆ
    }
    if scaler is not None and scaler.is_enabled():
        ckpt["scaler"] = scaler.state_dict()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)

    current = dataclasses.asdict(model.config)
    saved = ckpt["config"]
    shape_keys = ("vocab_size", "n_embd", "n_layer", "n_head", "block_size")
    mismatched = [k for k in shape_keys if current.get(k) != saved.get(k)]
    if mismatched:
        raise ValueError(
            f"checkpoint config mismatch on {mismatched}: saved={saved}, current={current}"
        )

    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt["config"]
