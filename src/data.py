# src/data.py
import os
import numpy as np
import torch


def get_batch_fn(data_dir: str, block_size: int, batch_size: int, split: str = "train"):
    """คืนค่าเป็นฟังก์ชัน get_batch() ที่สุ่ม batch ใหม่ทุกครั้งที่เรียก"""
    path = os.path.join(data_dir, f"{split}.bin")

    def get_batch():
        # เปิด memmap ใหม่ทุกครั้งที่เรียก ไม่เก็บ object ค้างไว้
        # (gotcha จาก nanoGPT: ถ้าเปิดค้างไว้ตัวเดียวมันจะค่อยๆกิน RAM เพิ่มเรื่อยๆ)
        data = np.memmap(path, dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
        )
        return x, y

    return get_batch