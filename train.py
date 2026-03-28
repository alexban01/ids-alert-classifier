from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import os

# ── Speed ────────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark        = True

MODEL        = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET      = "zeek_dataset.jsonl"       # train split from preprocess_zeek.py
EVAL_DATASET = "zeek_dataset_eval.jsonl"  # held-out eval split (source-stratified)
OUTPUT_DIR   = "./v7.1-ids-model"           # training checkpoints
ADAPTER_DIR  = "./v7.1-ids-lora-adapter"    # final adapter

# ── 4-bit quantization ──────────────────────────────────────────────────────
# QLoRA: 4-bit base model stays the same — adapter output is hardware-agnostic.
# The final adapter runs on RTX 3070 (8 GB) at inference time via 4-bit loading.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb_config, device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ── LoRA ─────────────────────────────────────────────────────────────────────
# r=16 (up from v4's r=8) — doubles adapter capacity for harder attack types.
# Negligible VRAM impact at inference (~20 MB more).
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",    # all attention
        "gate_proj", "up_proj", "down_proj",        # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── Dataset ──────────────────────────────────────────────────────────────────
# Train and eval are pre-split by preprocess_zeek.py using source-stratified
# sampling (10% per source/class bucket) so eval reflects real-world variety
# from all sources — not a random slice of the merged pool.
train_dataset = load_dataset("json", data_files=DATASET)["train"]
eval_dataset  = load_dataset("json", data_files=EVAL_DATASET)["train"]

# ── Training ─────────────────────────────────────────────────────────────────
# RTX 3070 (8 GB VRAM) — local machine.
# 4-bit weights dequantize to BF16 during the forward pass, spiking VRAM.
# batch_size=4 + accumulation=6 keeps effective batch=24 (same as 3090 config).
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        # ── Batch size ────────────────────────────────────────────────────
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=6,  # effective batch = 24
        optim="paged_adamw_8bit",
        # ── Precision ────────────────────────────────────────────────────
        gradient_checkpointing=True,
        bf16=True,
        # ── Schedule ─────────────────────────────────────────────────────
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 3},
        warmup_ratio=0.03,
        weight_decay=0.01,
        # ── Data loading ─────────────────────────────────────────────────
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        # ── Eval & saving ────────────────────────────────────────────────
        logging_steps=250,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=512,
    ),
)

trainer.train()

# ── Save ─────────────────────────────────────────────────────────────────────
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"\n✅ Training complete. Best adapter saved to {ADAPTER_DIR}")
print("   Download this directory to your local machine for inference/GGUF conversion.")
print(f"\n   Next: .venv/bin/python benchmark_realworld.py --regen")
