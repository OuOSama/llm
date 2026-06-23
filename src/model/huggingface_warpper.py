# src/model/huggingface_warpper.py

import torch
from typing import Optional, Tuple, Union
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.model.config import LLMConfig
from src.model.model import LLM


class LLM_HF_Config(PretrainedConfig):
    model_type = "yae_llm"

    def __init__(
        self,
        vocab_size=2000,
        block_size=128,
        n_layer=6,
        n_head=8,
        n_embd=384,
        dropout=0.1,
        rope_theta=10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.rope_theta = rope_theta


class HFLLMWrapper(PreTrainedModel, GenerationMixin):
    config_class = LLM_HF_Config
    supports_gradient_checkpointing = False

    def __init__(self, config: LLM_HF_Config):
        super().__init__(config)

        self.llm_config = LLMConfig(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
        )

        self.model = LLM(self.llm_config)
        self.post_init()

    def get_output_embeddings(self):
        return self.model.lm_head

    def get_input_embeddings(self):
        return self.model.wte

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.model.wte = new_embeddings

    def tie_weights(self, **kwargs):
        self.model.lm_head.weight = self.model.wte.weight

    def save_pretrained(self, *args, **kwargs):
        # ชั่วคราว untie weights ก่อน save ค่ะ~
        original_weight = self.model.lm_head.weight
        self.model.lm_head.weight = torch.nn.Parameter(
            self.model.wte.weight.detach().clone()
        )
        super().save_pretrained(*args, **kwargs)
        # retie กลับค่ะ~
        self.model.lm_head.weight = original_weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        logits, loss = self.model(input_ids, targets=labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
