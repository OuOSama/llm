# src/HF/model.py

import torch
import torch.nn as nn
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)


class CustomQwenParts(nn.Module):
    def __init__(self, vocab_size: int = 151936, n_embd: int = 768, n_head: int = 12):
        # แนะนำปรับ Vocab_size เป็น 151936 ตามมาตรฐานพจนานุกรมทองคำของ Qwen เลยค่ะ!
        super().__init__()

        # 1. ตั้งค่าคอนฟิกสไตล์ Qwen2 ครบวงจร
        self.config = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=n_embd,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            intermediate_size=n_embd * 4,
            hidden_act="silu",
            max_position_embeddings=2048,
        )

        # 2. สร้างอุปกรณ์คำนวณ RoPE สไตล์ Qwen (ต่างจาก LLaMA เล็กน้อยเรื่องสูตรเบส)
        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)

        # 3. หยิบชิ้นส่วนย่อยของ Qwen2 มาฟิวชั่นเอง!
        # ชิ้นส่วน A: แอตเทนชันเวอร์ชั่นมี QKV Bias ในตัว
        self.my_attn = Qwen2Attention(config=self.config, layer_idx=0)

        # ชิ้นส่วน B: MLP (SwiGLU สปีดรันความรู้)
        self.my_mlp = Qwen2MLP(config=self.config)

        # อุปกรณ์เสริม: ตัวคุม RMSNorm อันแสนประหยัดเมมโมรี่
        self.input_layernorm = Qwen2RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

        # หัวท้ายโมเดล
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.size()

        # แปลงคำศัพท์เป็นเวกเตอร์
        x = self.wte(input_ids)

        # 🔥 จุดสำคัญ: สร้างมุมหมุน RoPE สดๆ ตามความยาว T ของ Qwen
        # จะพ่นค่าออกมาเป็น (cos, sin) ไปป้อนอาหารให้แอตเทนชันสับรางต่อ
        position_ids = torch.arange(
            0, t, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)

        # --- ลูปต่อสายไฟใน Transformer Block แบบคุมสลักเองซับซ้อน ---

        # 1. พาร์ท Attention + Residual Connection
        residual = x
        hidden_states = self.input_layernorm(x)

        # ยัด position_embeddings ลงบล็อกแอตเทนชันของ Qwen
        attn_outputs = self.my_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,  # ใส่ None ได้เลยเพราะเค้าทำ Causal Mask ให้ข้างในอัตโนมัติแล้วค่ะ
        )
        x = residual + attn_outputs[0]

        # 2. พาร์ท SwiGLU MLP + Residual Connection
        residual = x
        hidden_states = self.post_attention_layernorm(x)
        mlp_outputs = self.my_mlp(hidden_states)
        x = residual + mlp_outputs

        # ส่งออกหัว Linear ปั๊มค่าทำนายคำศัพท์
        logits = self.lm_head(x)
        return logits


class QwenForTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = CustomQwenParts()  
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            # Shift logits และ labels สำหรับงาน Causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return {"loss": loss, "logits": logits} if loss is not None else logits


# Test Run สปีดรันเช็คความแรงของร่างทอง Qwen2!
if __name__ == "__main__":
    model = CustomQwenParts()
    dummy_input = torch.randint(0, 2000, (2, 32))  # Batch=2, Seq_len=32
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")  # Expected: [2, 32, 151936]

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
