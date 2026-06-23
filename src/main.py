import torch
from transformers import PreTrainedTokenizerFast
from src.model.huggingface_warpper import HFLLMWrapper

# 1. โหลด Tokenizer ตัวเดิมที่ใช้ตอนเทรน
tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json")
tokenizer.pad_token = "[PAD]"

# 2. โหลดโมเดลที่เทรนเสร็จแล้ว (มันจะโหลดไฟล์จากโฟลเดอร์ results ให้เอง)
model = HFLLMWrapper.from_pretrained("./results/checkpoint-4250")
model.eval()  # สั่งให้เป็นโหมดพร้อมตอบคำถาม

# 3. ลองป้อนคำถามให้โมเดล
prompt = "god"
inputs = tokenizer(prompt, return_tensors="pt")

# 4. ให้โมเดล Generate คำตอบออกมา
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=1.2,  # ให้โอกาสคำอื่นๆ บ้าง
        top_p=0.9,  # เลือกจากกลุ่มคำที่มีความน่าจะเป็นสูง
        repetition_penalty=2.0,  # สั่งห้ามซ้ำแบบเด็ดขาด!
        no_repeat_ngram_size=2,  # ห้ามพูด 2 คำซ้ำเดิมติดกันเด็ดขาด
    )

# 5. แปลงรหัสตัวเลขกลับมาเป็นข้อความที่อ่านรู้เรื่อง
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
