# src/prepare_data.py
import os
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = enc.n_vocab  # 50257 — ต้องใช้ค่านี้ใน LLMConfig ด้วย


def prepare(input_txt_path: str, out_dir: str, val_fraction: float = 0.001):
    with open(input_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = enc.encode_ordinary(text)  # ไม่แทรก special token
    ids = np.array(ids, dtype=np.uint16)  # gpt2 vocab < 65536 พอดี ใช้ uint16 ประหยัด disk

    n_val = max(1, int(len(ids) * val_fraction))
    val_ids, train_ids = ids[:n_val], ids[n_val:]

    os.makedirs(out_dir, exist_ok=True)
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))
    print(f"train tokens: {len(train_ids):,} | val tokens: {len(val_ids):,}")


if __name__ == "__main__":
    prepare("data/raw_corpus.txt", out_dir="data")