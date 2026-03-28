"""
benchmark.py — Three-way comparison: Vanilla vs v4 vs v6 on CICIDS2017.

Runs each model sequentially (loads, infers, unloads) to stay within 8 GB VRAM.
Produces individual reports + a side-by-side summary table at the end.
Results saved to benchmark_report.txt and benchmark_results.json.

Usage:
    .venv/bin/python benchmark.py
"""

import os
import json
import random
import torch
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (
    classification_report, confusion_matrix, matthews_corrcoef
)

from prompt_utils import SYSTEM_PROMPT, build_prompt, extract_verdict

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
BENCHMARK_CACHE = "benchmark_samples_v4.json"
REPORT_TXT      = "benchmark_report.txt"
RESULTS_JSON    = "benchmark_results.json"
MAX_NEW_TOKENS  = 80
BATCH_SIZE      = 8
SAMPLES_PER_CLASS_PER_FILE = 50
RANDOM_SEED     = 42

MODELS = [
    ("Vanilla Qwen2.5-1.5B",  BASE_MODEL,              False),
    ("v4 Fine-tuned",          "./v4-ids-lora-adapter", True),
    ("v6 Fine-tuned",          "./v6-ids-lora-adapter", True),
]

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

CICIDS_COLS = [
    "Protocol", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Label",
]

# Fix CICIDS2017 label encoding artifacts (Latin-1 read as UTF-8)
LABEL_FIXES = {
    "Web Attack \x96 Brute Force": "Web Attack - Brute Force",
    "Web Attack \x96 XSS":         "Web Attack - XSS",
    "Web Attack \x96 Sql Injection":"Web Attack - Sql Injection",
    "Web Attack â Brute Force":    "Web Attack - Brute Force",
    "Web Attack â XSS":            "Web Attack - XSS",
    "Web Attack â Sql Injection":  "Web Attack - Sql Injection",
}

def fix_label(label):
    label = label.strip()
    return LABEL_FIXES.get(label, label)

# ── Sample generation ─────────────────────────────────────────────────────────
def label_to_verdict(label):
    return "FALSE POSITIVE" if fix_label(label) == "BENIGN" else "ATTACK"

def generate_benchmark_samples():
    samples = []
    for fpath in CSV_FILES:
        if not os.path.exists(fpath):
            print(f"[SKIP] Not found: {fpath}")
            continue
        print(f"[LOAD] {fpath}")
        df = pd.read_csv(fpath, low_memory=False, encoding="latin-1")
        df.columns = df.columns.str.strip()
        df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()

        avail = [c for c in CICIDS_COLS if c in df.columns]
        if "Label" not in df.columns:
            print(f"  [SKIP] No Label column")
            continue

        df["Label"] = df["Label"].apply(fix_label)
        benign  = df[df["Label"] == "BENIGN"]
        attacks = df[df["Label"] != "BENIGN"]
        n_benign  = min(SAMPLES_PER_CLASS_PER_FILE, len(benign))
        n_attacks = min(SAMPLES_PER_CLASS_PER_FILE, len(attacks))

        parts = []
        if n_benign  > 0: parts.append(benign.sample(n_benign,   random_state=RANDOM_SEED))
        if n_attacks > 0: parts.append(attacks.sample(n_attacks, random_state=RANDOM_SEED))
        sampled = pd.concat(parts)

        for _, row in sampled.iterrows():
            proto_num = str(int(float(row["Protocol"]))) if "Protocol" in avail else "unknown"
            dur_us    = row.get("Flow Duration", 0)
            try:
                duration = str(float(dur_us) / 1e6)
            except (ValueError, TypeError):
                duration = "0"

            orig_pkts  = str(row["Total Fwd Packets"])           if "Total Fwd Packets"           in avail else "0"
            resp_pkts  = str(row["Total Backward Packets"])      if "Total Backward Packets"      in avail else "0"
            orig_bytes = str(row["Total Length of Fwd Packets"]) if "Total Length of Fwd Packets" in avail else "0"
            resp_bytes = str(row["Total Length of Bwd Packets"]) if "Total Length of Bwd Packets" in avail else "0"
            conn_state = "-"

            prompt = build_prompt(proto_num, duration, orig_pkts, resp_pkts,
                                  orig_bytes, resp_bytes, conn_state, "-")
            samples.append({
                "prompt":       prompt,
                "ground_truth": label_to_verdict(row["Label"]),
                "source_file":  os.path.basename(fpath),
                "raw_label":    fix_label(row["Label"]),
            })
        print(f"  → {n_benign} benign + {n_attacks} attacks sampled")

    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    with open(BENCHMARK_CACHE, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\n✅ {len(samples)} benchmark samples saved to {BENCHMARK_CACHE}\n")
    return samples

# ── Dataset / inference ───────────────────────────────────────────────────────
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

def run_batched_inference(model, samples, label):
    dataset    = PromptDataset(samples)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )
    preds    = []
    unknowns = 0
    total    = len(samples)

    print(f"\nRunning inference: {label}")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch     = {k: v.to("cuda") for k, v in batch.items()}
            input_len = batch["input_ids"].shape[1]
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
            print(f"  {min((i+1)*BATCH_SIZE, total)}/{total}...", end="\r")

    print()
    return preds, unknowns

# ── Reporting ─────────────────────────────────────────────────────────────────
def compute_metrics(preds, samples, unknowns):
    truths     = [s["ground_truth"] for s in samples]
    labels     = ["ATTACK", "FALSE POSITIVE"]
    attack_idx = [i for i, t in enumerate(truths) if t == "ATTACK"]
    benign_idx = [i for i, t in enumerate(truths) if t == "FALSE POSITIVE"]

    acc        = sum(t == p for t, p in zip(truths, preds)) / len(truths)
    atk_recall = sum(preds[i] == "ATTACK"         for i in attack_idx) / max(len(attack_idx), 1)
    ben_recall = sum(preds[i] == "FALSE POSITIVE" for i in benign_idx) / max(len(benign_idx), 1)
    fmt_fail   = unknowns / len(truths)

    # MCC: handles class imbalance correctly — ranges from -1 (perfect inverse)
    # to +1 (perfect), 0 = random. Better single metric than accuracy for IDS.
    mcc_preds  = [1 if p == "ATTACK" else 0 for p in preds]
    mcc_truths = [1 if t == "ATTACK" else 0 for t in truths]
    mcc        = matthews_corrcoef(mcc_truths, mcc_preds)

    return {
        "accuracy":    acc,
        "atk_recall":  atk_recall,
        "ben_recall":  ben_recall,
        "fmt_fail":    fmt_fail,
        "mcc":         mcc,
        "truths":      truths,
        "preds":       preds,
    }

def print_report(preds, samples, label, unknowns, out_lines):
    truths = [s["ground_truth"] for s in samples]
    labels = ["ATTACK", "FALSE POSITIVE"]
    mcc    = matthews_corrcoef(
        [1 if t == "ATTACK" else 0 for t in truths],
        [1 if p == "ATTACK" else 0 for p in preds],
    )

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  MODEL : {label}")
    lines.append(f"  Total : {len(samples)} samples")
    lines.append(f"  Format failures : {unknowns} ({100*unknowns/len(samples):.1f}%)")
    lines.append(f"  MCC             : {mcc:+.4f}  (range -1 to +1, higher is better)")
    lines.append(f"{'='*60}")
    lines.append(classification_report(truths, preds, labels=labels, zero_division=0))

    lines.append("Confusion Matrix  (rows = actual, cols = predicted)")
    lines.append(f"{'':22s} {'ATTACK':>10} {'FALSE POSITIVE':>15}")
    cm = confusion_matrix(truths, preds, labels=labels)
    for row_label, row in zip(labels, cm):
        total_row = row.sum()
        pct = [f"{100*v/total_row:.0f}%" for v in row]
        lines.append(f"  {row_label:20s} {row[0]:>6} ({pct[0]:>4})  {row[1]:>6} ({pct[1]:>4})")

    lines.append(f"\n--- Per attack type breakdown ---")
    for raw in sorted(set(s["raw_label"] for s in samples)):
        idx     = [i for i, s in enumerate(samples) if s["raw_label"] == raw]
        correct = sum(truths[i] == preds[i] for i in idx)
        lines.append(f"  {raw:44s} {correct:>3}/{len(idx):<3} ({100*correct/len(idx):>3.0f}%)")

    lines.append(f"\n--- Per CSV file breakdown ---")
    for fpath in sorted(set(s["source_file"] for s in samples)):
        idx     = [i for i, s in enumerate(samples) if s["source_file"] == fpath]
        correct = sum(truths[i] == preds[i] for i in idx)
        lines.append(f"  {os.path.basename(fpath):55s} {correct:>3}/{len(idx):<3} ({100*correct/len(idx):>3.0f}%)")

    for line in lines:
        print(line)
    out_lines.extend(lines)

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(path, is_finetuned):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if is_finetuned:
        print(f"Loading base + LoRA adapter from {path} ...")
        base  = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map="cuda"
        )
        model = PeftModel.from_pretrained(base, path)
    else:
        print(f"Loading vanilla model: {path} ...")
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="cuda"
        )
    model.eval()
    return model

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    if os.path.exists(BENCHMARK_CACHE):
        print(f"[CACHE] Loading existing samples from {BENCHMARK_CACHE}")
        with open(BENCHMARK_CACHE) as f:
            samples = json.load(f)
        # Fix any encoding artifacts in cached samples
        for s in samples:
            s["raw_label"] = fix_label(s["raw_label"])
        print(f"  {len(samples)} samples loaded")
    else:
        samples = generate_benchmark_samples()

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    out_lines = [
        f"BENCHMARK REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Samples: {len(samples)} | Dataset: CICIDS2017 | Seed: {RANDOM_SEED}",
    ]

    results     = []
    json_output = {"timestamp": datetime.now().isoformat(), "samples": len(samples), "models": []}

    for label, path, is_finetuned in MODELS:
        if is_finetuned and not os.path.isdir(path):
            print(f"\n[SKIP] {label} — adapter not found at {path}")
            continue

        model           = load_model(path, is_finetuned)
        preds, unknowns = run_batched_inference(model, samples, label)
        print_report(preds, samples, label, unknowns, out_lines)
        m               = compute_metrics(preds, samples, unknowns)
        results.append((label, m))

        json_output["models"].append({
            "label":      label,
            "accuracy":   round(m["accuracy"],   4),
            "atk_recall": round(m["atk_recall"], 4),
            "ben_recall": round(m["ben_recall"], 4),
            "fmt_fail":   round(m["fmt_fail"],   4),
            "mcc":        round(m["mcc"],         4),
        })

        del model
        torch.cuda.empty_cache()

    # ── Comparison table ──────────────────────────────────────────────────────
    if len(results) > 1:
        header = [
            f"\n{'='*80}",
            f"  COMPARISON SUMMARY",
            f"{'='*80}",
            f"  {'Model':<26} {'Accuracy':>9} {'Atk Recall':>11} {'FP Recall':>10} {'MCC':>7} {'Fmt Fail':>9}",
            f"  {'-'*26} {'-'*9} {'-'*11} {'-'*10} {'-'*7} {'-'*9}",
        ]
        rows = []
        for label, m in results:
            rows.append(
                f"  {label:<26} {m['accuracy']:>9.1%} {m['atk_recall']:>11.1%}"
                f" {m['ben_recall']:>10.1%} {m['mcc']:>+7.3f} {m['fmt_fail']:>9.1%}"
            )

        # Delta row: v6 vs v4 (if both present)
        labels = [r[0] for r in results]
        if "v4 Fine-tuned" in labels and "v6 Fine-tuned" in labels:
            v4 = dict(results)[  "v4 Fine-tuned"]
            v6 = dict(results)[  "v6 Fine-tuned"]
            delta = (
                f"  {'v6 delta vs v4':<26}"
                f" {v6['accuracy']-v4['accuracy']:>+9.1%}"
                f" {v6['atk_recall']-v4['atk_recall']:>+11.1%}"
                f" {v6['ben_recall']-v4['ben_recall']:>+10.1%}"
                f" {v6['mcc']-v4['mcc']:>+7.3f}"
                f" {v6['fmt_fail']-v4['fmt_fail']:>+9.1%}"
            )
            rows.append(f"  {'-'*26} {'-'*9} {'-'*11} {'-'*10} {'-'*7} {'-'*9}")
            rows.append(delta)

        footer = [
            f"{'='*80}",
            "",
            "  Atk Recall = % of actual attacks correctly caught (sensitivity)",
            "  FP Recall  = % of benign flows correctly identified (higher = fewer false alarms)",
            "  MCC        = Matthews Correlation Coefficient — best single metric for",
            "               imbalanced binary classification (range -1 to +1)",
            "  Fmt Fail   = % of outputs where no VERDICT line was found",
        ]

        for line in header + rows + footer:
            print(line)
        out_lines.extend(header + rows + footer)

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(REPORT_TXT, "w") as f:
        f.write("\n".join(out_lines))
    with open(RESULTS_JSON, "w") as f:
        json.dump(json_output, f, indent=2)

    print(f"\n✅ Report saved to {REPORT_TXT}")
    print(f"✅ Results saved to {RESULTS_JSON}")
