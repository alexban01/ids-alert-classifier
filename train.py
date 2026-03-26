from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import os

# ── Speed ────────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark        = True

MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET     = "zeek_dataset.jsonl"      # output from preprocess_zeek.py
OUTPUT_DIR  = "./v5-ids-model"          # training checkpoints
ADAPTER_DIR = "./v5-ids-lora-adapter"   # final adapter

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
dataset = load_dataset("json", data_files=DATASET)["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# ── Training ─────────────────────────────────────────────────────────────────
# Targeting RTX 5090 (32 GB VRAM) on RunPod.
# Key differences from v4 (RTX 3070, 8 GB):
#   - batch_size 2→16 (8x fewer forward passes, same effective batch)
#   - gradient_checkpointing OFF (saves ~30% compute overhead)
#   - eval + load_best_model_at_end re-enabled (impossible on 8 GB)
#   - 3 epochs with cosine_with_restarts (3 cycles, 1 per epoch)
#   - LR 2e-4 (up from 5e-5) — QLoRA paper sweet spot
#   - dataloader_num_workers=4 (RunPod uses standard Python, no forkserver issue)
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        # ── Batch size ────────────────────────────────────────────────────
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,  # effective batch = 16
        optim="paged_adamw_8bit",
        # ── Precision ────────────────────────────────────────────────────
        gradient_checkpointing=True,    # safety margin — RunPod eats ~9 GB
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
        logging_steps=100,
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
