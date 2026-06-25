import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import os

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--runpod", action="store_true",
                    help="Use RunPod RTX 5090 settings (batch=24, no grad checkpointing)")
parser.add_argument("--epochs", type=int, default=2,
                    help="Training epochs (default 2). load_best_model_at_end keeps the "
                         "best-eval_loss checkpoint, so 2 epochs ≈ 3 in quality but ~33%% cheaper.")
parser.add_argument("--no-pack", action="store_true",
                    help="Disable sequence packing. Packing (default ON) concatenates the "
                         "variable-length samples (mean 296 tok) into full max_length sequences. "
                         "Measured on the real set: 241k seqs -> 165k packed (0.69x steps), ~20%% "
                         "fewer tokens/epoch. The run trains on the full sequence either way "
                         "(no completion-only masking), so packing does not change the loss objective.")
parser.add_argument("--eval-subset", type=int, default=6000,
                    help="Cap eval set to this many samples (0 = use all). Eval runs every epoch "
                         "purely to pick the best checkpoint by eval_loss; 6k is plenty and saves "
                         "most of the eval-pass cost vs the full 31k held-out set.")
args = parser.parse_args()
RUNPOD  = args.runpod
EPOCHS  = args.epochs
PACKING = not args.no_pack

if RUNPOD:
    BATCH                = 24
    GRAD_ACCUM           = 1      # effective batch = 24
    GRAD_CHECKPOINTING   = False  # 32 GB has headroom even at max_length=512
    PIN_MEMORY           = True
    NUM_WORKERS          = 0      # CUDA+fork unstable on Linux regardless of hardware; bottleneck is GPU not JSONL loading
    MAX_LENGTH           = 512
else:
    BATCH                = 4
    GRAD_ACCUM           = 6      # effective batch = 24 (6×4)
    GRAD_CHECKPOINTING   = True   # required for 8 GB VRAM; 1024 tokens needs smaller batch
    PIN_MEMORY           = False
    NUM_WORKERS          = 0      # CUDA+fork unstable on local Linux. fork() copies the parent's CUDA context into worker processes — those handles are invalid in the child, causing deadlocks or corruption. spawn would fix it but adds complexity; workers=0 is simpler since the bottleneck is the GPU, not JSONL loading.
    MAX_LENGTH           = 512

print(f"Target: {'RunPod RTX 5090' if RUNPOD else 'Local RTX 3070'}  "
      f"| batch={BATCH}  accum={GRAD_ACCUM}  effective={BATCH*GRAD_ACCUM}  "
      f"max_length={MAX_LENGTH}  epochs={EPOCHS}  packing={PACKING}")

# ── Speed ────────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark        = True

MODEL        = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET      = "zeek_dataset.jsonl"       # train split from preprocess_zeek.py
EVAL_DATASET = "zeek_dataset_eval.jsonl"  # held-out eval split (source-stratified)
OUTPUT_DIR   = "./v12-ids-model"           # training checkpoints
ADAPTER_DIR  = "./v12-ids-lora-adapter"    # final adapter

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
# r=32 (up from v8's r=16) — more capacity for port-aware patterns (Credential Access,
# Defense Evasion). Training VRAM delta: ~72 MB (weights + gradients); optimizer states
# are paged to CPU so those don't count. Inference delta: ~40 MB.
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
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

# Eval is only used to pick the best checkpoint by eval_loss — a stable metric that
# does not need the full 31k held-out set. Subsample (the eval file is already
# source-stratified and shuffled by preprocess_zeek.py) to cut eval-pass cost.
if args.eval_subset and len(eval_dataset) > args.eval_subset:
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(args.eval_subset))
    print(f"Eval subsampled to {len(eval_dataset)} samples (--eval-subset {args.eval_subset})")

# ── Training ─────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        # ── Batch size ────────────────────────────────────────────────────
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        optim="paged_adamw_8bit",
        # ── Precision ────────────────────────────────────────────────────
        gradient_checkpointing=GRAD_CHECKPOINTING,
        bf16=True,
        # ── Throughput ───────────────────────────────────────────────────
        # Packing fills each max_length sequence with multiple short samples,
        # cutting opt-steps 241k->165k seqs (0.69x) and ~20% tokens/epoch.
        packing=PACKING,
        # ── Schedule ─────────────────────────────────────────────────────
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": EPOCHS},  # one cosine restart per epoch
        warmup_ratio=0.03,
        weight_decay=0.01,
        # ── Data loading ─────────────────────────────────────────────────
        dataloader_pin_memory=PIN_MEMORY,
        dataloader_num_workers=NUM_WORKERS,
        # ── Eval & saving ────────────────────────────────────────────────
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,            # keep best + last; avoid disk churn from every epoch
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=MAX_LENGTH,   # v10: extended for multi-log prompts (http/dns/ssl/behavior context)
    ),
)

trainer.train()

# ── Save ─────────────────────────────────────────────────────────────────────
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"\n✅ Training complete. Best adapter saved to {ADAPTER_DIR}")
print("   Download this directory to your local machine for inference/GGUF conversion.")
print(f"\n   Next: .venv/bin/python benchmark_realworld.py --regen")
print(f"   OOD check: .venv/bin/python benchmark_realworld.py --ood-only  (Botnet-3 Kelihos)")
