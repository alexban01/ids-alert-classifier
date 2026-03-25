import os
import json
import random
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ───────────────────────────────────────────────────────────────────
BASE_MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_MODEL = "./v4-ids-lora-adapter"
BENCHMARK_CACHE = "benchmark_samples.json"
MAX_NEW_TOKENS  = 80
BATCH_SIZE      = 8     # lower to 4 if OOM
SAMPLES_PER_CLASS_PER_FILE = 50
RANDOM_SEED     = 42

CSV_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

FEATURE_COLS = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "Average Packet Size",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
]

SYSTEM_PROMPT = (
    "You are a network security analyst. "
    "Always respond with VERDICT: <ATTACK or FALSE POSITIVE> on the first line, "
    "followed by REASON: <brief explanation>."
)

# ── Dataset generation ────────────────────────────────────────────────────────
def label_to_verdict(label):
    return "FALSE POSITIVE" if label.strip() == "BENIGN" else "ATTACK"

def build_prompt(row, avail_cols):
    lines = ["Analyze this network flow and classify it as ATTACK or FALSE POSITIVE.\n"]
    lines += [f"  {c}: {row[c]}" for c in avail_cols]
    return "\n".join(lines)

def generate_benchmark_samples():
    samples = []
    for fpath in CSV_FILES:
        if not os.path.exists(fpath):
            print(f"[SKIP] Not found: {fpath}")
            continue
        print(f"[LOAD] {fpath}")
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()
        avail = [c for c in FEATURE_COLS if c in df.columns]

        benign  = df[df["Label"] == "BENIGN"]
        attacks = df[df["Label"] != "BENIGN"]
        n_benign  = min(SAMPLES_PER_CLASS_PER_FILE, len(benign))
        n_attacks = min(SAMPLES_PER_CLASS_PER_FILE, len(attacks))

        parts = []
        if n_benign  > 0: parts.append(benign.sample(n_benign,   random_state=RANDOM_SEED))
        if n_attacks > 0: parts.append(attacks.sample(n_attacks, random_state=RANDOM_SEED))
        sampled = pd.concat(parts)

        for _, row in sampled.iterrows():
            samples.append({
                "prompt":       build_prompt(row, avail),
                "ground_truth": label_to_verdict(row["Label"]),
                "source_file":  os.path.basename(fpath),
                "raw_label":    row["Label"].strip(),
            })
        print(f"  → {n_benign} benign + {n_attacks} attacks sampled")

    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    with open(BENCHMARK_CACHE, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\n✅ {len(samples)} benchmark samples saved to {BENCHMARK_CACHE}\n")
    return samples

# ── PyTorch Dataset ───────────────────────────────────────────────────────────
# tokenizer is set as a module-level variable before DataLoader is created
_tokenizer = None

class PromptDataset(Dataset):
    def __init__(self, samples):
        self.texts = [
            _tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": s["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for s in samples
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def collate_fn(batch):
    return _tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

# ── Inference ─────────────────────────────────────────────────────────────────
def extract_verdict(output):
    for line in output.upper().splitlines():
        if "VERDICT:" in line:
            if "FALSE POSITIVE" in line:
                return "FALSE POSITIVE"
            if "ATTACK" in line:
                return "ATTACK"
    return "UNKNOWN"

def run_batched_inference(model, samples, label):
    dataset    = PromptDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    preds    = []
    unknowns = 0
    total    = len(samples)

    print(f"\nRunning inference: {label}")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch      = {k: v.to("cuda") for k, v in batch.items()}
            input_len  = batch["input_ids"].shape[1]
            out = model.generate(
                **batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=_tokenizer.pad_token_id,
            )
            for seq in out:
                text    = _tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
                verdict = extract_verdict(text)
                if verdict == "UNKNOWN":
                    unknowns += 1
                preds.append(verdict)

            done = min((i + 1) * BATCH_SIZE, total)
            print(f"  {done}/{total}...", end="\r")

    print()
    return preds, unknowns

# ── Report ────────────────────────────────────────────────────────────────────
def print_report(preds, samples, label, unknowns):
    truths = [s["ground_truth"] for s in samples]
    labels = ["ATTACK", "FALSE POSITIVE"]

    print(f"\n{'='*60}")
    print(f"  MODEL : {label}")
    print(f"  Total : {len(samples)} samples")
    print(f"  Format failures: {unknowns} ({100*unknowns/len(samples):.1f}%)")
    print(f"{'='*60}")
    print(classification_report(truths, preds, labels=labels, zero_division=0))

    print("Confusion Matrix  (rows = actual, cols = predicted)")
    print(f"{'':22s} {'ATTACK':>10} {'FALSE POSITIVE':>15}")
    cm = confusion_matrix(truths, preds, labels=labels)
    for row_label, row in zip(labels, cm):
        print(f"  {row_label:20s} {row[0]:>10} {row[1]:>15}")

    print(f"\n--- Per attack type breakdown ---")
    for raw in sorted(set(s["raw_label"] for s in samples)):
        idx     = [i for i, s in enumerate(samples) if s["raw_label"] == raw]
        correct = sum(truths[i] == preds[i] for i in idx)
        print(f"  {raw:42s} {correct}/{len(idx)} ({100*correct/len(idx):.0f}%)")

    return truths

# ── Load model helper ─────────────────────────────────────────────────────────
def load_model(path, is_finetuned=False):
    """
    Vanilla Qwen  → load with 4-bit BitsAndBytes quantization.

    Fine-tuned model → two cases:
      1. Clean adapter dir (adapter_config.json exists):
         Load base model + PeftModel.from_pretrained.
      2. Legacy PeftModel state dict (v2, no adapter_config.json):
         Load base model, wrap with LoRA config, inject LoRA weights manually.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if is_finetuned:
        print(f"Loading base model + LoRA weights from {path} ...")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="cuda",
        )

        if os.path.isfile(os.path.join(path, "adapter_config.json")):
            # ── Clean adapter saved by PeftModel.save_pretrained ──────────
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, path)
            print("  Loaded adapter via PeftModel.from_pretrained")
        else:
            # ── Legacy: v2 checkpoint with PeftModel-wrapped key names ────
            # Manually apply same LoRA config and inject saved weights.
            lora_config = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(base, lora_config)
            saved = load_file(os.path.join(path, "model.safetensors"))
            lora_state = {}
            for k, v in saved.items():
                if "lora_" in k:
                    lora_state["base_model.model." + k] = v
            info = model.load_state_dict(lora_state, strict=False)
            print(f"  Loaded {len(lora_state)} LoRA tensors (legacy path)")
            assert not info.unexpected_keys, (
                f"Key mismatch — {len(info.unexpected_keys)} LoRA tensors "
                f"did not match. First few: {info.unexpected_keys[:3]}"
            )
    else:
        print(f"Loading vanilla model: {path} ...")
        model = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=bnb_config,
            device_map="cuda",
        )

    model.eval()
    return model

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    # Load or generate benchmark samples
    if os.path.exists(BENCHMARK_CACHE):
        print(f"[CACHE] Loading existing samples from {BENCHMARK_CACHE}")
        with open(BENCHMARK_CACHE) as f:
            samples = json.load(f)
        print(f"  {len(samples)} samples loaded")
    else:
        samples = generate_benchmark_samples()

    # Tokenizer is shared — both models use the same Qwen2.5 vocabulary
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    all_results = {}

    # ── Vanilla Qwen ──────────────────────────────────────────────────────────
    model = load_model(BASE_MODEL, is_finetuned=False)
    vanilla_preds, vanilla_unknowns = run_batched_inference(
        model, samples, "Vanilla Qwen2.5-1.5B-Instruct"
    )
    print_report(vanilla_preds, samples, "Vanilla Qwen2.5-1.5B-Instruct", vanilla_unknowns)
    all_results["vanilla"] = vanilla_preds
    del model
    torch.cuda.empty_cache()

    # ── Fine-tuned model ──────────────────────────────────────────────────────
    model = load_model(FINETUNED_MODEL, is_finetuned=True)
    finetuned_preds, finetuned_unknowns = run_batched_inference(
        model, samples, "Fine-tuned Qwen2.5-1.5B (v2)"
    )
    print_report(finetuned_preds, samples, "Fine-tuned Qwen2.5-1.5B (v2)", finetuned_unknowns)
    all_results["finetuned"] = finetuned_preds
    del model
    torch.cuda.empty_cache()

    # ── Head-to-head summary ──────────────────────────────────────────────────
    truths     = [s["ground_truth"] for s in samples]
    attack_idx = [i for i, t in enumerate(truths) if t == "ATTACK"]
    benign_idx = [i for i, t in enumerate(truths) if t == "FALSE POSITIVE"]

    def accuracy(p):      return sum(t == x for t, x in zip(truths, p)) / len(truths)
    def fmt_fail(p):      return sum(x == "UNKNOWN" for x in p) / len(truths)
    def atk_recall(p):    return sum(p[i] == "ATTACK"         for i in attack_idx) / len(attack_idx)
    def benign_recall(p): return sum(p[i] == "FALSE POSITIVE" for i in benign_idx) / len(benign_idx)

    v = all_results["vanilla"]
    f = all_results["finetuned"]

    print(f"\n{'='*60}")
    print(f"  HEAD-TO-HEAD SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':40s} {'Vanilla':>10} {'Fine-tuned':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Overall accuracy':40s} {accuracy(v):>9.1%} {accuracy(f):>9.1%}")
    print(f"  {'Attack recall (catch rate)':40s} {atk_recall(v):>9.1%} {atk_recall(f):>9.1%}")
    print(f"  {'Benign recall (false pos rate)':40s} {benign_recall(v):>9.1%} {benign_recall(f):>9.1%}")
    print(f"  {'Format failure rate':40s} {fmt_fail(v):>9.1%} {fmt_fail(f):>9.1%}")
    print(f"{'='*60}")
