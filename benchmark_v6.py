"""
benchmark_v6.py — v4 vs v6 fine-tuned model comparison on CICIDS2017.

Usage:
    .venv/bin/python benchmark_v6.py
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
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

from prompt_utils import SYSTEM_PROMPT, build_prompt, extract_verdict

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
BENCHMARK_CACHE = "results/benchmark_samples_v4.json"
REPORT_TXT      = "results/benchmark_v6_report.txt"
RESULTS_JSON    = "results/benchmark_v6_results.json"
MAX_NEW_TOKENS  = 80
BATCH_SIZE      = 8
SAMPLES_PER_CLASS_PER_FILE = 50
RANDOM_SEED     = 42

MODELS = [
    ("v4 Fine-tuned",   "./v4-ids-lora-adapter"),
    ("v6 Fine-tuned",   "./v6-ids-lora-adapter"),
    ("v7.1 Fine-tuned", "./v7.1-ids-lora-adapter"),
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

LABEL_FIXES = {
    "Web Attack \x96 Brute Force":  "Web Attack - Brute Force",
    "Web Attack \x96 XSS":          "Web Attack - XSS",
    "Web Attack \x96 Sql Injection": "Web Attack - Sql Injection",
    "Web Attack â Brute Force":     "Web Attack - Brute Force",
    "Web Attack â XSS":             "Web Attack - XSS",
    "Web Attack â Sql Injection":   "Web Attack - Sql Injection",
}

def fix_label(label):
    return LABEL_FIXES.get(label.strip(), label.strip())

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

        if "Label" not in df.columns:
            continue

        df["Label"] = df["Label"].apply(fix_label)
        avail    = [c for c in CICIDS_COLS if c in df.columns]
        benign   = df[df["Label"] == "BENIGN"]
        attacks  = df[df["Label"] != "BENIGN"]
        n_benign  = min(SAMPLES_PER_CLASS_PER_FILE, len(benign))
        n_attacks = min(SAMPLES_PER_CLASS_PER_FILE, len(attacks))

        parts = []
        if n_benign  > 0: parts.append(benign.sample(n_benign,   random_state=RANDOM_SEED))
        if n_attacks > 0: parts.append(attacks.sample(n_attacks, random_state=RANDOM_SEED))
        sampled = pd.concat(parts)

        for _, row in sampled.iterrows():
            proto_num = str(int(float(row["Protocol"]))) if "Protocol" in avail else "unknown"
            try:
                duration = str(float(row.get("Flow Duration", 0)) / 1e6)
            except (ValueError, TypeError):
                duration = "0"

            orig_pkts  = str(row["Total Fwd Packets"])           if "Total Fwd Packets"           in avail else "0"
            resp_pkts  = str(row["Total Backward Packets"])      if "Total Backward Packets"      in avail else "0"
            orig_bytes = str(row["Total Length of Fwd Packets"]) if "Total Length of Fwd Packets" in avail else "0"
            resp_bytes = str(row["Total Length of Bwd Packets"]) if "Total Length of Bwd Packets" in avail else "0"

            samples.append({
                "prompt":       build_prompt(proto_num, duration, orig_pkts, resp_pkts,
                                            orig_bytes, resp_bytes, "-", "-"),
                "ground_truth": label_to_verdict(row["Label"]),
                "source_file":  os.path.basename(fpath),
                "raw_label":    fix_label(row["Label"]),
            })
        print(f"  → {n_benign} benign + {n_attacks} attacks")

    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    with open(BENCHMARK_CACHE, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\n✅ {len(samples)} samples saved to {BENCHMARK_CACHE}\n")
    return samples

# ── Dataset / inference ───────────────────────────────────────────────────────
_tokenizer = None

class PromptDataset(Dataset):
    def __init__(self, samples):
        self.texts = [
            _tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user",   "content": s["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for s in samples
        ]
    def __len__(self):         return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx]

def collate_fn(batch):
    return _tokenizer(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512)

def run_inference(model, samples, label):
    loader   = DataLoader(PromptDataset(samples), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True,
                          collate_fn=collate_fn)
    preds    = []
    unknowns = 0
    print(f"\nRunning inference: {label}")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch     = {k: v.to("cuda") for k, v in batch.items()}
            input_len = batch["input_ids"].shape[1]
            out = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=_tokenizer.pad_token_id)
            for seq in out:
                text    = _tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
                verdict = extract_verdict(text)
                if verdict == "UNKNOWN":
                    unknowns += 1
                preds.append(verdict)
            print(f"  {min((i+1)*BATCH_SIZE, len(samples))}/{len(samples)}...", end="\r")
    print()
    return preds, unknowns

# ── Reporting ─────────────────────────────────────────────────────────────────
def compute_metrics(preds, samples, unknowns):
    truths     = [s["ground_truth"] for s in samples]
    attack_idx = [i for i, t in enumerate(truths) if t == "ATTACK"]
    benign_idx = [i for i, t in enumerate(truths) if t == "FALSE POSITIVE"]
    mcc = matthews_corrcoef(
        [1 if t == "ATTACK" else 0 for t in truths],
        [1 if p == "ATTACK" else 0 for p in preds],
    )
    return {
        "accuracy":    sum(t == p for t, p in zip(truths, preds)) / len(truths),
        "atk_recall":  sum(preds[i] == "ATTACK"         for i in attack_idx) / max(len(attack_idx), 1),
        "ben_recall":  sum(preds[i] == "FALSE POSITIVE" for i in benign_idx) / max(len(benign_idx), 1),
        "fmt_fail":    unknowns / len(truths),
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
    lines = [
        f"\n{'='*62}",
        f"  MODEL : {label}",
        f"  Total : {len(samples)} samples",
        f"  Format failures : {unknowns} ({100*unknowns/len(samples):.1f}%)",
        f"  MCC             : {mcc:+.4f}  (range -1 to +1, higher is better)",
        f"{'='*62}",
        classification_report(truths, preds, labels=labels, zero_division=0),
        "Confusion Matrix  (rows = actual, cols = predicted)",
        f"{'':22s} {'ATTACK':>10} {'FALSE POSITIVE':>15}",
    ]
    cm = confusion_matrix(truths, preds, labels=labels)
    for row_label, row in zip(labels, cm):
        pct = [f"{100*v/row.sum():.0f}%" for v in row]
        lines.append(f"  {row_label:20s} {row[0]:>6} ({pct[0]:>4})  {row[1]:>6} ({pct[1]:>4})")

    lines.append(f"\n--- Per attack type ---")
    for raw in sorted(set(s["raw_label"] for s in samples)):
        idx     = [i for i, s in enumerate(samples) if s["raw_label"] == raw]
        correct = sum(truths[i] == preds[i] for i in idx)
        lines.append(f"  {raw:44s} {correct:>3}/{len(idx):<3} ({100*correct/len(idx):>3.0f}%)")

    lines.append(f"\n--- Per CSV file ---")
    for fpath in sorted(set(s["source_file"] for s in samples)):
        idx     = [i for i, s in enumerate(samples) if s["source_file"] == fpath]
        correct = sum(truths[i] == preds[i] for i in idx)
        lines.append(f"  {fpath:55s} {correct:>3}/{len(idx):<3} ({100*correct/len(idx):>3.0f}%)")

    for line in lines:
        print(line)
    out_lines.extend(lines)

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(adapter_path):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)
    print(f"Loading base + adapter from {adapter_path} ...")
    base  = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                                  device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    if os.path.exists(BENCHMARK_CACHE):
        print(f"[CACHE] Loading {BENCHMARK_CACHE}")
        with open(BENCHMARK_CACHE) as f:
            samples = json.load(f)
        for s in samples:
            s["raw_label"] = fix_label(s["raw_label"])
        print(f"  {len(samples)} samples loaded")
    else:
        samples = generate_benchmark_samples()

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    out_lines   = [f"v4 vs v6 BENCHMARK — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                   f"Samples: {len(samples)} | Dataset: CICIDS2017 | Seed: {RANDOM_SEED}"]
    results     = []
    json_output = {"timestamp": datetime.now().isoformat(), "samples": len(samples), "models": []}

    for label, adapter_path in MODELS:
        if not os.path.isdir(adapter_path):
            print(f"\n[SKIP] {label} — not found at {adapter_path}")
            continue

        model           = load_model(adapter_path)
        preds, unknowns = run_inference(model, samples, label)
        print_report(preds, samples, label, unknowns, out_lines)
        m               = compute_metrics(preds, samples, unknowns)
        results.append((label, m))
        json_output["models"].append({
            "label": label, "accuracy": round(m["accuracy"], 4),
            "atk_recall": round(m["atk_recall"], 4),
            "ben_recall": round(m["ben_recall"], 4),
            "fmt_fail":   round(m["fmt_fail"],   4),
            "mcc":        round(m["mcc"],         4),
        })

        del model
        torch.cuda.empty_cache()

    # ── Comparison table ──────────────────────────────────────────────────────
    if len(results) == 2:
        (l4, m4), (l6, m6) = results
        summary = [
            f"\n{'='*72}",
            f"  v4 vs v6 COMPARISON",
            f"{'='*72}",
            f"  {'Model':<20} {'Accuracy':>9} {'Atk Recall':>11} {'FP Recall':>10} {'MCC':>7} {'Fmt Fail':>9}",
            f"  {'-'*20} {'-'*9} {'-'*11} {'-'*10} {'-'*7} {'-'*9}",
            f"  {l4:<20} {m4['accuracy']:>9.1%} {m4['atk_recall']:>11.1%} {m4['ben_recall']:>10.1%} {m4['mcc']:>+7.3f} {m4['fmt_fail']:>9.1%}",
            f"  {l6:<20} {m6['accuracy']:>9.1%} {m6['atk_recall']:>11.1%} {m6['ben_recall']:>10.1%} {m6['mcc']:>+7.3f} {m6['fmt_fail']:>9.1%}",
            f"  {'-'*20} {'-'*9} {'-'*11} {'-'*10} {'-'*7} {'-'*9}",
            f"  {'delta (v6 - v4)':<20}"
            f" {m6['accuracy']-m4['accuracy']:>+9.1%}"
            f" {m6['atk_recall']-m4['atk_recall']:>+11.1%}"
            f" {m6['ben_recall']-m4['ben_recall']:>+10.1%}"
            f" {m6['mcc']-m4['mcc']:>+7.3f}"
            f" {m6['fmt_fail']-m4['fmt_fail']:>+9.1%}",
            f"{'='*72}",
            "",
            "  Atk Recall = % of actual attacks correctly caught (sensitivity)",
            "  FP Recall  = % of benign flows correctly identified (fewer false alarms)",
            "  MCC        = Matthews Correlation Coefficient (-1 to +1, higher is better)",
            "  Fmt Fail   = % of outputs with no VERDICT line",
        ]
        for line in summary:
            print(line)
        out_lines.extend(summary)

    with open(REPORT_TXT, "w") as f:
        f.write("\n".join(out_lines))
    with open(RESULTS_JSON, "w") as f:
        json.dump(json_output, f, indent=2)

    print(f"\n✅ Report saved to {REPORT_TXT}")
    print(f"✅ Results saved to {RESULTS_JSON}")
