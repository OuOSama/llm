# src/count_parameters.py

from model import LLM
from generate import load_model_for_inference


model = load_model_for_inference("checkpoints/best.pt")


def count_parameters(model: LLM):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-" * 40)
    print(f"🔥 Total Parameters: {total_params:,}")
    print(f"🎯 Trainable Parameters: {trainable_params:,}")
    print("-" * 40)


count_parameters(model)
