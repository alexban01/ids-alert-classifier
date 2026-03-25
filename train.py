from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import os

# ── Speed: enable TF32 on Ampere+ (RTX 3070 = Ampere) ───────────────────────
# TF32 uses 19-bit floats for matmuls — same range as fp32 but 8x faster.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark        = True

MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET     = "ids_dataset.jsonl"       # output from preprocess.py
OUTPUT_DIR  = "./v3-ids-model"          # training checkpoints
ADAPTER_DIR = "./v3-ids-lora-adapter"   # final adapter

# ── 4-bit quantization ────────────────────────────────────────────────────────
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

# ── LoRA ──────────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",    # all attention
        "gate_proj", "up_proj", "down_proj",        # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files=DATASET)["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# ── Training ──────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        # ── Batch size ────────────────────────────────────────────────────
        # paged_adamw_8bit stores optimizer states in 8-bit (~halves VRAM
        # for Adam's m/v buffers), freeing enough room for batch_size=2.
        # That means 2x fewer forward passes than batch_size=1.
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch = 16
        optim="paged_adamw_8bit",       # 8-bit optimizer — key VRAM saver
        # ── Checkpointing & precision ─────────────────────────────────────
        gradient_checkpointing=True,
        bf16=True,
        # ── Schedule ──────────────────────────────────────────────────────
        num_train_epochs=3,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        # ── Data loading (CPU helps HERE — async prefetch) ────────────────
        dataloader_pin_memory=True,     # page-locked CPU RAM → faster transfer
        dataloader_num_workers=0,       # must be 0 on Python 3.14 (forkserver)
        # ── Logging & saving ──────────────────────────────────────────────
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=512,
    ),
)

trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
# Use trainer.model (the PeftModel) to save adapter files properly.
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"\n✅ Training complete. Adapter saved to {ADAPTER_DIR}")
print("   Check that adapter_config.json exists in the output directory.")
print("   If it does, benchmark.py can load it via PeftModel.from_pretrained().")
print("   If only model.safetensors exists, benchmark.py will use the LoRA key extraction fallback.")
