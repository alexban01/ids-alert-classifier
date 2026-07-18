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
parser.add_argument("--ibm", action="store_true",
                    help="Use IBM Cloud L40S 48GB settings — same config as --runpod "
                         "(batch=24, no grad checkpointing); only the run.json target differs.")
parser.add_argument("--epochs", type=int, default=2,
                    help="Training epochs (default 2). load_best_model_at_end keeps the "
                         "best-eval_loss checkpoint, so 2 epochs ≈ 3 in quality but ~33%% cheaper.")
parser.add_argument("--no-pack", action="store_true",
                    help="Disable sequence packing. Packing (default ON) concatenates the "
                         "variable-length samples (mean 296 tok) into full max_length sequences. "
                         "Measured on the real set: 241k seqs -> 165k packed (0.69x steps), ~20%% "
                         "fewer tokens/epoch. Loss MASKING is unaffected (full-sequence loss either "
                         "way, no completion-only masking) -- but packing is NOT loss-neutral: TRL's "
                         "default 'bfd' packing strategy requires a flash-attention variant to keep "
                         "packed samples from attending to each other; this project has no flash-attn "
                         "installed and no attn_implementation set (SDPA/eager default), so packed "
                         "sequences DO let self-attention leak across unrelated samples sharing a "
                         "512-token slot (~1.46 samples/seq on average). --no-pack removes that "
                         "confound at the ~20%% throughput cost.")
parser.add_argument("--eval-subset", type=int, default=6000,
                    help="Cap eval set to this many samples (0 = use all). Eval runs every epoch "
                         "purely to pick the best checkpoint by eval_loss; 6k is plenty and saves "
                         "most of the eval-pass cost vs the full 31k held-out set.")
parser.add_argument("--save-steps", type=int, default=0,
                    help="Save (and eval) every N optimizer steps instead of every epoch. "
                         "0 (default) = epoch-based, unchanged. >0 = step-based, so a run can be "
                         "interrupted mid-epoch and resumed. Intended for local RTX 3070 runs; "
                         "load_best_model_at_end forces eval at each save point, so a too-small N "
                         "adds eval overhead. eval_steps is pinned equal to save_steps.")
parser.add_argument("--resume", nargs="?", const=True, default=False,
                    help="Resume training. Bare --resume continues from the latest checkpoint in "
                         "OUTPUT_DIR; --resume <path> continues from a specific checkpoint dir. "
                         "Restores model + optimizer + LR scheduler + step counter. Requires the "
                         "dataset/batch/packing/epochs to be unchanged from the interrupted run.")
parser.add_argument("--tag", type=str, default="v13",
                    help="Run name -> output dirs models/<tag>-ids-model (checkpoints) and "
                         "models/<tag>-ids-lora-adapter (final adapter). Bump per variant, e.g. "
                         "v13 / v13.1 / v13.2, so parallel experiments don't clobber each other.")
parser.add_argument("--flash-attn", action="store_true",
                    help="Load the base model with attn_implementation='flash_attention_2'. "
                         "Required for TRL's bfd packing to actually isolate packed samples "
                         "from each other (SDPA/eager let them cross-attend — the v12.2 vs "
                         "v13.1 leak finding). Needs the flash-attn package installed; "
                         "errors out early if missing. Default (no flag) is unchanged (SDPA).")
parser.add_argument("--completion-only", action="store_true",
                    help="v14a: mask loss to the assistant turn only. Maps the messages-format "
                         "dataset to prompt/completion at load time (file on disk unchanged, so "
                         "the run.json dataset hash stays valid); TRL then auto-enables "
                         "completion_only_loss. All gradient lands on 'VERDICT: <X>' instead of "
                         "~95%% of it modeling the ~300-token prompt. eval_loss is NOT comparable "
                         "to full-sequence runs — compare by MCC only.")
parser.add_argument("--lora-r", type=int, default=16,
                    help="LoRA rank (alpha is always 2r). Default 16 (V10/v13 setting); "
                         "32 restores the v11 recipe.")
parser.add_argument("--dataset", type=str, default="zeek_dataset.jsonl",
                    help="Train file (e.g. a preprocess_downsample.py output for a data-volume "
                         "ablation). Eval file is unaffected by this flag -- always "
                         "zeek_dataset_eval.jsonl, so eval_loss stays comparable across runs.")
args = parser.parse_args()
RUNPOD  = args.runpod or args.ibm   # --ibm: identical config, different run.json target
TARGET  = ("IBM Cloud L40S" if args.ibm else
           "RunPod RTX 5090" if args.runpod else "Local RTX 3070")
EPOCHS  = args.epochs
PACKING = not args.no_pack

# Step-based saving lets a run be interrupted mid-epoch and resumed (see --resume).
# load_best_model_at_end requires eval_strategy to match save_strategy, so eval is
# pinned to the same cadence. 0 keeps the original epoch-based behaviour (RunPod default).
if args.save_steps and args.save_steps > 0:
    SAVE_STRATEGY, EVAL_STRATEGY = "steps", "steps"
    SAVE_STEPS = EVAL_STEPS = args.save_steps
else:
    SAVE_STRATEGY, EVAL_STRATEGY = "epoch", "epoch"
    SAVE_STEPS = EVAL_STEPS = None

if RUNPOD:
    # L40S 48GB OOMs at batch=24 with packing (every seq a full 512 tokens, no grad
    # checkpointing, fp32 CE upcast over the 152k vocab ~7.5GB) — 12x2 keeps the
    # effective batch at 24 (cross-run comparability) and halves peak memory.
    BATCH                = 12 if args.ibm else 24
    GRAD_ACCUM           = 2 if args.ibm else 1   # effective batch = 24
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

print(f"Target: {TARGET}  "
      f"| batch={BATCH}  accum={GRAD_ACCUM}  effective={BATCH*GRAD_ACCUM}  "
      f"max_length={MAX_LENGTH}  epochs={EPOCHS}  packing={PACKING}")

# ── Speed ────────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark        = True

MODEL        = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET      = args.dataset               # train split from preprocess_zeek.py (or a downsample)
EVAL_DATASET = "zeek_dataset_eval.jsonl"  # held-out eval split (source-stratified) -- fixed,
                                           # so eval_loss stays comparable across data-volume runs
OUTPUT_DIR   = f"./models/{args.tag}-ids-model"           # training checkpoints
ADAPTER_DIR  = f"./models/{args.tag}-ids-lora-adapter"    # final adapter

# ── 4-bit quantization ──────────────────────────────────────────────────────
# QLoRA: 4-bit base model stays the same — adapter output is hardware-agnostic.
# The final adapter runs on RTX 3070 (8 GB) at inference time via 4-bit loading.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# --flash-attn: fail fast if the package is missing (a mid-run crash after model
# download is the expensive way to find out). Default path passes no
# attn_implementation — byte-identical to historic behaviour (resolves to sdpa).
model_kwargs = {}
if args.flash_attn:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        raise SystemExit("--flash-attn requires the flash-attn package: "
                         ".venv/bin/pip install flash-attn (see STATE.md Next steps #4)")
    model_kwargs["attn_implementation"] = "flash_attention_2"

model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb_config, device_map="cuda", **model_kwargs
)
ATTN_IMPLEMENTATION = model.config._attn_implementation  # resolved value, recorded in run.json
print(f"attn_implementation: {ATTN_IMPLEMENTATION}")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ── LoRA ─────────────────────────────────────────────────────────────────────
# Default r=16/alpha=32 (V10's setting, reverted from V11/V12's r=32/alpha=64). V10 is
# still the best-ever Win7AD-1 OOD result (87.1%) and was never matched at r=32 — the
# extra capacity may just memorize ID patterns harder without helping (or hurting) OOD
# transfer, same overfitting shape as the epoch-2-hurts-OOD finding. v13 tests this.
# --lora-r 32 restores the v11 setting (alpha always = 2r) — used by v15 (v11 recipe
# + v14.x training fixes, quality-first).
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=2 * args.lora_r,
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

# --completion-only (v14a): messages -> {prompt, completion}. TRL detects the
# prompt/completion form and sets completion_only_loss=True automatically, masking
# loss to the assistant turn ('VERDICT: <X>') in both padded and packed collators.
if args.completion_only:
    def to_prompt_completion(ex):
        return {"prompt": ex["messages"][:-1], "completion": ex["messages"][-1:]}
    train_dataset = train_dataset.map(to_prompt_completion, remove_columns=["messages"])
    eval_dataset  = eval_dataset.map(to_prompt_completion,  remove_columns=["messages"])
    print("completion-only loss: dataset mapped to prompt/completion form")

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
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        eval_steps=EVAL_STEPS,         # None when epoch-based; ignored by HF in that case
        save_steps=SAVE_STEPS,
        save_total_limit=2,            # keep best + last; avoid disk churn from every epoch
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=MAX_LENGTH,   # v10: extended for multi-log prompts (http/dns/ssl/behavior context)
    ),
)

# ── Per-epoch adapter snapshots (for model soup) ─────────────────────────────
# save_total_limit=2 rotates full checkpoints away, so epoch-boundary weights are
# lost — which blocks souping epochs together. This dumps just the LoRA adapter
# (a few MB, no optimizer state) at every epoch end into epoch-N/, outside the
# rotation, regardless of --save-steps.
from transformers import TrainerCallback

class SaveEpochAdapter(TrainerCallback):
    def on_epoch_end(self, targs, state, control, model=None, **kwargs):
        out = os.path.join(targs.output_dir, f"epoch-{round(state.epoch)}")
        model.save_pretrained(out)
        print(f"[soup] adapter snapshot saved to {out}")

trainer.add_callback(SaveEpochAdapter())

# ── Resume ───────────────────────────────────────────────────────────────────
# --resume <path> uses that checkpoint; bare --resume picks the latest in OUTPUT_DIR.
# Falls back to a fresh run (with a notice) if asked to resume but nothing is there.
resume_from = None
if args.resume:
    if isinstance(args.resume, str):
        resume_from = args.resume
    else:
        from transformers.trainer_utils import get_last_checkpoint
        last = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
        if last:
            resume_from = last
        else:
            print(f"[WARN] --resume given but no checkpoint found in {OUTPUT_DIR}; starting fresh.")
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")

train_result = trainer.train(resume_from_checkpoint=resume_from)

# ── Save ─────────────────────────────────────────────────────────────────────
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"\n✅ Training complete. Best adapter saved to {ADAPTER_DIR}")
print("   Download this directory to your local machine for inference/GGUF conversion.")

# ── Run manifest (run.json + EXPERIMENTS.md) ──────────────────────────────────
# Self-describing provenance: hyperparams + dataset content hash + best eval_loss.
# Wrapped so bookkeeping can never fail a completed training run.
try:
    from ids.run_manifest import write_run_manifest
    write_run_manifest(
        ADAPTER_DIR,
        base_model=MODEL,
        target=TARGET,
        hyperparams={
            "epochs":          trainer.args.num_train_epochs,
            "packing":         getattr(trainer.args, "packing", PACKING),
            "attn_implementation": ATTN_IMPLEMENTATION,
            "completion_only":  args.completion_only,
            "batch":           trainer.args.per_device_train_batch_size,
            "grad_accum":      trainer.args.gradient_accumulation_steps,
            "effective_batch": BATCH * GRAD_ACCUM,
            "max_length":      MAX_LENGTH,
            "learning_rate":   trainer.args.learning_rate,
            "lora_r":          lora_config.r,
            "lora_alpha":      lora_config.lora_alpha,
            "lora_dropout":    lora_config.lora_dropout,
            "eval_subset":     args.eval_subset,
        },
        train_file=DATASET,
        eval_loss=trainer.state.best_metric,
        train_runtime_s=train_result.metrics.get("train_runtime"),
    )
    print(f"   📋 run.json + EXPERIMENTS.md updated")
    # Copy run.json into each epoch-N/ snapshot so resolve_system_prompt() serves
    # the right prompt (verdict-only for --no-reason runs) when benchmarking them.
    import glob, shutil
    for snap in glob.glob(os.path.join(OUTPUT_DIR, "epoch-*")):
        shutil.copy(os.path.join(ADAPTER_DIR, "run.json"), snap)
except Exception as e:
    print(f"[WARN] could not write run manifest: {e}")
print(f"\n   Next: .venv/bin/python benchmark_realworld.py --regen")
print(f"   OOD check: .venv/bin/python benchmark_realworld.py --ood-only  (Botnet-3 Kelihos)")
