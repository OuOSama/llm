# src/train/train.py

from transformers import PreTrainedTokenizerFast
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from src.model.huggingface_warpper import HFLLMWrapper, LLM_HF_Config

from src.train.data import get_dataset

tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json")
tokenizer.pad_token = "[PAD]"


def train():
    config = LLM_HF_Config(
        vocab_size=32000, block_size=512, n_layer=8, n_head=8, n_embd=512
    )

    model = HFLLMWrapper(config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        learning_rate=5e-4,
        weight_decay=0.01,
        gradient_checkpointing=False,
        bf16=True,
        max_grad_norm=1.0,
        warmup_steps=0.4,
        # 📊 TensorBoard
        report_to="tensorboard",
        logging_steps=10,
        # 💾 Checkpoint ทุก 250 steps
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,  # เก็บแค่ 3 อันล่าสุดไม่งั้น disk เต็มค่ะ~
        # 🏆 Best model — eval ทุก 250 steps เช่นกัน
        eval_strategy="steps",
        eval_steps=250,
        load_best_model_at_end=True,  # โหลด best กลับมาตอนจบค่ะ!
        metric_for_best_model="eval_loss",  # วัดจาก loss ค่ะ~
        greater_is_better=False,  # loss ยิ่งน้อยยิ่งดีค่ะ!
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=get_dataset(),
        eval_dataset=get_dataset(split="validation"),  # ← ต้องมี val set ด้วยนะคะ!!
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
        ], 
    )

    trainer.train()


if __name__ == "__main__":
    train()
