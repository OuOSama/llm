# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LLMConfig:
    vocab_size: int  # ขนาดของพจนานุกรมคำศัพท์ (Vocabulary)
    block_size: int  # ความยาวบริบทสูงสุดที่โมเดลรับได้ (Context Window / Max Sequence Length)
    n_layer: int = 12  # จำนวนชั้นประมวลผล (Transformer Blocks)
    n_head: int = 12  # จำนวนหัวในการทำ Attention
    n_embd: int = 768  # ขนาดมิติเวกเตอร์ของคำ (Embedding Dimension)
    dropout: float = 0.1
    rope_theta: float = 10000.0  # ค่าฐานความถี่สำหรับคำนวณตำแหน่งแบบ RoPE


# ---------------------------------------------------------------------------
# RoPE helpers: ระบบจัดการตำแหน่งของคำด้วยการหมุนเวียนเวกเตอร์
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """หั่นมิติสุดท้ายออกเป็น 2 ครึ่ง (x1, x2) แล้วสลับฝั่งพร้อมติดลบตัวหลัง: -> (-x2, x1)"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """ทำการหมุนมุมเวกเตอร์ q หรือ k ตามตำแหน่งของคำในประโยค
    x:   (Batch, n_head, T_seq, head_dim)
    cos: (T_seq, head_dim) -> จะถูกขยายมิติเพื่อบอร์ดแคสท์
    sin: (T_seq, head_dim)
    """
    # ขยายมิติจาก (T, hd) -> (1, 1, T, hd) เพื่อให้คูณกระจายเข้ามิติ Batch และ n_head ได้อัตโนมัติ
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    # สูตรรูปแบบ Matrix Rotation: (x * cos) + (rotate_half(x) * sin)
    return (x * cos) + (rotate_half(x) * sin)


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    device=None,
):
    """
    อัปเกรดเป็น Long Context Scaling สไตล์ Qwen2 / Llama3
    scaling_factor: ยิ่งค่าเยอะ ยิ่งรองรับ Context ที่ยาวขึ้น (เช่น 4.0 หรือ 8.0)
    """
    assert head_dim % 2 == 0, "head_dim ต้องหาร 2 ลงตัว"

    # 1. คำนวณความถี่พื้นฐาน
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    # 2. 🔥 ไส้ในที่แท้จริงของการอัปเกรด:
    # หาร inv_freq ด้วย scaling_factor เพื่อ "ยืด" ค่าความถี่ให้ครอบคลุมตำแหน่งที่ไกลขึ้น
    # วิธีนี้คือ Linear Interpolation แบบง่ายที่สุดที่ทำให้โมเดลอ่านประโยคยาวๆ ได้
    inv_freq = inv_freq / scaling_factor

    # 3. คำนวณเหมือนเดิม
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    return emb.cos(), emb.sin()


# ---------------------------------------------------------------------------
# Components: บล็อกย่อยของโมเดล
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # ใช้ Linear ชั้นเดียวแปลงอินพุตออกเป็น Q, K, V พร้อมกัน (มิติขยายเป็น 3 เท่า)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # ตัวบีบมิติกลับหลังจากทำ Attention เสร็จสิ้น
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = x.size()  # Batch, Sequence Length, Channels (n_embd)

        # 1. แตกข้อมูลออกเป็นสามเส้นแล้วแบ่งสัดส่วนให้เป็น Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2. ปรับรูปร่าง Matrix แยกตามหัวประมวลผล: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. หมุนเวกเตอร์ Q และ K เพื่อฝังข้อมูลตำแหน่งสัมพัทธ์ (RoPE)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 4. เรียกใช้ Fused Kernel คำนวณ FlashAttention ตัวแรงออโต้ พร้อมใส่ Causal Mask ปิดตาดูอนาคต
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # 5. สลับมิติกลับและมัดรวมหัวประมวลผลเข้าด้วยกัน: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 6. ฉายสเปกตรัมกลับเข้ามิติดั้งเดิมและทำ Dropout
        y = self.resid_drop(self.c_proj(y))
        return y


class MLP(nn.Module):
    """บล็อกคำนวณเครือข่ายประสาทเทียมสไตล์โมเดิร์น (SwiGLU Architecture)"""

    def __init__(self, config: LLMConfig):
        super().__init__()
        # คำนวณมิติซ่อนเร้นตามมาตรฐาน SwiGLU (ปัดเศษคูณด้วยประมาณ 2/3 ของ 4x มิติตั้งต้น)
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # เส้นคุมประตู (Gate)
        self.w2 = nn.Linear(
            hidden_dim, config.n_embd, bias=False
        )  # เส้นบีบมิติออก (Down-projection)
        self.w3 = nn.Linear(
            config.n_embd, hidden_dim, bias=False
        )  # เส้นรับค่าประมวลผล (Value)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        # โลจิก SwiGLU: เอาผลลัพธ์เส้น w1 ผ่าน SiLU (Swish) แล้วนำมาคูณดوتกับเส้น w3 ก่อนยิงออกทาง w2
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """บล็อกโครงสร้าง Transformer 1 ชั้น จัดวางเลย์เอาต์แบบ Pre-LayerNorm"""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        # ชั้นที่ 1: บีบกระจายนอร์มข้อมูล -> คำนวณความสัมพันธ์คำ (Attention) -> บวกทางลัด (Residual Connection)
        x = x + self.attn(self.ln_1(x), cos, sin)
        # ชั้นที่ 2: บีบกระจายนอร์มข้อมูล -> คำนวณความรู้จำคำ (MLP) -> บวกทางลัด (Residual Connection)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Main Model: ตัวควบคุมการประมวลผลโครงข่ายหลัก
# ---------------------------------------------------------------------------


class LLM(nn.Module):
    # กำหนด Type Hint ล่วงหน้าเพื่อให้ระบบจำแนกประเภทข้อมูลของ IDE ไม่เอ๋อและขึ้นเส้นแดง
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # ชั้นแปลงไอดีคำศัพท์ให้กลายเป็นเวกเตอร์ความหมายข้อมูล (Token Embedding)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Precompute คำนวณแผนผัง RoPE Cache เก็บไว้ที่ตัวแม่ตั้งแต่ตอนสร้างโมเดล
        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope_cache(
            head_dim=head_dim, max_seq_len=config.block_size, theta=config.rope_theta
        )
        # ลงทะเบียนเป็น Buffer เพื่อให้ย้ายไป GPU อัตโนมัติเวลาสั่ง model.to(device) โดยไม่นับเป็น Weight Parameter
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

        # ต่อบล็อกประมวลผลลึกเข้าไปจำนวน n_layer ชั้น
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # นอร์มปิดท้ายโครงสร้างหลัก
        self.ln_f = nn.RMSNorm(config.n_embd)
        # ชั้น Linear สำหรับพ่นแจกโอกาสคำศัพท์เพื่อทำนายคำถัดไป
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # เทคนิค Weight Tying: ผูกน้ำหนักขาออกเข้ากับขาเข้า เพื่อประหยัดเมมโมรี่และเพิ่มประสิทธิภาพการเรียนรู้
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()  # ขนาด Batch และความยาว Sequence ปัจจุบันที่โมเดลกำลังประมวลผล
        x = self.drop(self.wte(idx))

        # ดึงช่วงตารางแคช RoPE ออกมาตามความยาว T ในแบทช์ และแคสต์ประเภทข้อมูลให้แมตช์กัน
        cos_slot = self.cos_cached[:T].to(x.dtype)
        sin_slot = self.sin_cached[:T].to(x.dtype)

        # ส่งสัญญาณไหลทะลวงผ่านบล็อกสถาปัตยกรรมทีละชั้น
        for block in self.h:
            x = block(x, cos_slot, sin_slot)

        x = self.ln_f(x)
        logits = self.lm_head(
            x
        )  # ผลลัพธ์คะแนนความน่าจะเป็นของแต่ละคำศัพท์ขนาด (B, T, vocab_size)

        loss = None
        if targets is not None:
            # ยุบขนาดมิติ Matrix ให้เหลือแนวเดียวเพื่อส่งเข้าคำนวณ Cross Entropy Loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        """ฟังก์ชัน Autoregressive สำหรับสร้างประโยคข้อความแบบสุ่มเดาคำถัดไปทีละคำ"""
        for _ in range(max_new_tokens):
            # หั่นบริบทข้อมูลอินพุตไม่ให้ยาวเกินกว่าขอบเขตสูงสุดที่แคช RoPE รองรับ
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            # ดึงคะแนนทำนายของตำแหน่งสุดท้ายมาหารด้วยอุณหภูมิ (Temperature) เพื่อคุมความสร้างสรรค์ของคำ
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # คัดกรองตัวเลือกคำศัพท์ เลือกเอาเฉพาะตัวตึงท็อปๆ เพื่อป้องกันโมเดลพูดเพ้อเจ้อ
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < thresh, torch.full_like(logits, -float("inf")), logits
                )

            # แปลงคะแนนดิบให้เป็นเปอร์เซ็นต์ความน่าจะเป็นแล้วสุ่มจับไอดีคำถัดไปมาต่อท้ายสายอาร์เรย์
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
