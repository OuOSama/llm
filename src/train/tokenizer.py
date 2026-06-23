# src/train/tokenizer.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def train_my_tokenizer(files, vocab_size=32000):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer.train(files, trainer)
    tokenizer.save("data/tokenizer.json")
    print("Tokenizer Done!")


if __name__ == "__main__":
    train_my_tokenizer(["data/raw_corpus.txt"])
