"""Shared model-loading and chat-templating helpers for HF/PEFT inference.

Centralises the 4-bit base load + LoRA adapter attach + tokenizer setup +
Qwen2.5 chat-template application that every inference entry point
(benchmark_realworld, benchmark_v6, classify_conn_log, classify_weird_log,
compare_binetflow) previously reimplemented.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from prompt_utils import SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_tokenizer(base_model=BASE_MODEL, padding_side="left"):
    """Load the base tokenizer with a guaranteed pad token (left-padded for batch gen)."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side=padding_side)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lora_model(adapter_path, base_model=BASE_MODEL):
    """Load the 4-bit NF4 base model and attach a LoRA adapter; returns an eval model."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


def chat_text(tokenizer, prompt):
    """Render a single user prompt into the Qwen2.5 chat template (generation-ready)."""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
