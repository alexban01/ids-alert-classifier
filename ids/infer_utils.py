"""Shared model-loading and chat-templating helpers for HF/PEFT inference.

Centralises the 4-bit base load + LoRA adapter attach + tokenizer setup +
Qwen2.5 chat-template application that every inference entry point
(benchmark_realworld, benchmark_v6, classify_conn_log, classify_weird_log,
compare_binetflow) previously reimplemented.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from ids.prompt_utils import SYSTEM_PROMPT, SYSTEM_PROMPT_VERDICT_ONLY

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


def chat_text(tokenizer, prompt, system_prompt=SYSTEM_PROMPT):
    """Render a single user prompt into the Qwen2.5 chat template (generation-ready).

    system_prompt defaults to the standard VERDICT+REASON prompt; pass
    SYSTEM_PROMPT_VERDICT_ONLY (or use resolve_system_prompt) when serving a
    model trained with preprocess_zeek.py --no-reason.
    """
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )


def resolve_system_prompt(adapter_path):
    """Pick the system prompt matching how an adapter was trained.

    Reads <adapter_path>/run.json (written by train.py). A run trained with
    --no-reason records dataset.reason == False and must be prompted with the
    verdict-only system prompt — otherwise the system instruction asks for a
    REASON the model never learned to emit, silently confounding results.

    Returns (system_prompt, run_dict). run_dict is None when no run.json exists,
    in which case the default (reason-on) prompt is used — matching the historic
    behavior of adapters trained before run manifests existed.
    """
    from ids.run_manifest import read_json  # local import: keep infer_utils dep-light
    run = read_json(os.path.join(adapter_path, "run.json"))
    if not run:
        return SYSTEM_PROMPT, None
    reason = (run.get("dataset") or {}).get("reason")
    system_prompt = SYSTEM_PROMPT_VERDICT_ONLY if reason is False else SYSTEM_PROMPT
    return system_prompt, run
