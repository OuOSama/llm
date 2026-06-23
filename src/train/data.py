# src/train/data.py

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


def get_dataset(split: str = "train"):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json")
    tokenizer.pad_token = "[PAD]"

    dataset = load_dataset(
        "text",
        data_files={"train": "data/raw_corpus.txt"},
        split="train",  
    )

    # แบ่ง 95% train / 5% validation ค่ะ!
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=512, padding="max_length"
        )

    tokenized = split_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    if split == "train":
        return tokenized["train"]
    elif split == "validation":
        return tokenized["test"]  # HuggingFace เรียก split นี้ว่า "test" ค่ะ~
    else:
        raise ValueError(
            f"Unknown split: '{split}' — ใช้ได้แค่ 'train' หรือ 'validation' นะคะ!!"
        )
