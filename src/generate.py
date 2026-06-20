# src/generate.py

import torch
import tiktoken

from model import LLM, LLMConfig, device

enc = tiktoken.get_encoding("gpt2")


def load_model_for_inference(checkpoint_path: str) -> LLM:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = LLMConfig(
        **ckpt["config"]
    )  # rebuild config จาก dict ที่เซฟไว้ตรงๆ ไม่ต้องพิมพ์ใหม่
    model = LLM(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()  # สำคัญมาก! ปิด dropout ตอน inference
    return model


def generate_text(
    model: LLM,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    ids = enc.encode_ordinary(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    )
    return enc.decode(out[0].tolist())


if __name__ == "__main__":
    model = load_model_for_inference("checkpoints/final.pt")
    text = generate_text(
        model,
        prompt="OuOSama: Hello! Who are you?\nKING RICHARD II: I am the King, and your name is",
    )
    print(text)
