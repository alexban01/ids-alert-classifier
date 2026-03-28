"""
benchmark_realworld.py — v4 vs v6 on real-world Zeek data.

Uses native Zeek conn.log sources (IoT-23, CTU-13, UWF-ZeekData24, CTU-Normal)
with authentic conn_state values and protocol names — unlike benchmark_v6.py
which uses CICIDS2017 (CICFlowMeter CSVs with synthetic field mapping).

This is the meaningful test: v4 had ~95% FP rate on real Zeek logs. v6 was
trained specifically to fix that by adding UWF-ZeekData24 + CTU-Normal.

Usage:
    .venv/bin/python benchmark_realworld.py [--regen]

    --regen   Force regeneration of the sample cache even if it exists.
"""

import os
import sys
import json
import random
import tarfile
import glob
import torch
import pandas as pd
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

from prompt_utils import SYSTEM_PROMPT, build_prompt, extract_verdict

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_FILE   = "benchmark_realworld_cache.json"
REPORT_TXT   = "benchmark_realworld_report.txt"
RESULTS_JSON = "benchmark_realworld_results.json"
MAX_NEW_TOKENS = 80
BATCH_SIZE     = 8
CAP            = 300      # max samples per (source, class)
RANDOM_SEED    = 42

MODELS = [
    # ("v4 Fine-tuned",   "./v4-ids-lora-adapter"),
    # ("v6 Fine-tuned",   "./v6-ids-lora-adapter"),
    ("v7.1 Fine-tuned", "./v7.1-ids-lora-adapter"),
]

DATASETS = {
    "iot23":      "datasets/iot-23/iot_23_datasets_small.tar.gz",
    "ctu13":      "datasets/ctu-13/CTU-13-Dataset.tar.bz2",
    "uwf":        "datasets/uwf-zeekdata24/",
    "ctu_normal": "datasets/ctu-normal/",
}

# ── Sample helpers ──────────────────────────────────────────────────────────────
def make_sample(proto, duration, orig_pkts, resp_pkts,
                orig_bytes, resp_bytes, conn_state,
                ground_truth, source, raw_label, service="-"):
    return {
        "prompt":       build_prompt(proto, duration, orig_pkts, resp_pkts,
                                     orig_bytes, resp_bytes, conn_state, service),
        "ground_truth": ground_truth,
        "source":       source,
        "raw_label":    raw_label,
    }

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_iot23(archive_path):
    """Native Zeek conn.log.labeled from IoT-23 tar.gz.

    Last tab field bundles: tunnel_parents label detailed-label (space-separated).
    Detailed label examples: C&C, DDoS, Okiru, PartOfAHorizontalPortScan, etc.
    """
    if not os.path.isfile(archive_path):
        print(f"[SKIP] IoT-23 not found: {archive_path}")
        return []

    print(f"[IoT-23] Opening {archive_path} ...")
    buckets = defaultdict(list)   # key: ground_truth

    with tarfile.open(archive_path, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.name.endswith("conn.log.labeled") and m.isfile()]
        print(f"  {len(members)} conn.log.labeled file(s)")

        for member in members:
            f = tf.extractfile(member)
            if f is None:
                continue
            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 21:
                    continue

                last = parts[-1]
                if "Malicious" in last:
                    verdict = "ATTACK"
                elif "Benign" in last:
                    verdict = "FALSE POSITIVE"
                else:
                    continue

                if len(buckets[verdict]) >= CAP:
                    continue

                # Extract detailed label (3rd space-token in last field)
                sub = last.split()
                detailed = sub[2] if len(sub) >= 3 else sub[-1] if sub else "-"
                raw_label = detailed if verdict == "ATTACK" else "Benign"

                try:
                    buckets[verdict].append(make_sample(
                        proto      = parts[6],
                        duration   = parts[8],
                        orig_pkts  = parts[16],
                        resp_pkts  = parts[18],
                        orig_bytes = parts[9],
                        resp_bytes = parts[10],
                        conn_state = parts[11],
                        ground_truth = verdict,
                        source     = "iot23",
                        raw_label  = raw_label,
                        service    = parts[7],
                    ))
                except IndexError:
                    continue

                # Stop early if both buckets are full
                if all(len(v) >= CAP for v in buckets.values()):
                    break

    atk = len(buckets["ATTACK"])
    ben = len(buckets["FALSE POSITIVE"])
    print(f"  IoT-23: {atk} attacks, {ben} benign")
    return buckets["ATTACK"] + buckets["FALSE POSITIVE"]


def load_ctu13(archive_path):
    """CTU-13 binetflow CSV from tar.bz2.

    Has TotPkts only (split 50/50 between orig/resp).
    Label: contains 'Botnet' → ATTACK, 'Normal' → FP, 'Background' → skip.
    """
    if not os.path.isfile(archive_path):
        print(f"[SKIP] CTU-13 not found: {archive_path}")
        return []

    print(f"[CTU-13] Opening {archive_path} ...")
    buckets = defaultdict(list)

    with tarfile.open(archive_path, "r:bz2") as tf:
        members = [m for m in tf.getmembers()
                   if m.name.endswith(".binetflow") and m.isfile()]
        print(f"  {len(members)} binetflow file(s)")

        for member in members:
            if all(len(v) >= CAP for v in [buckets["ATTACK"], buckets["FALSE POSITIVE"]]):
                break
            f = tf.extractfile(member)
            if f is None:
                continue
            header = None
            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if header is None:
                    header = [c.strip() for c in line.split(",")]
                    continue
                parts = line.split(",")
                if len(parts) < len(header):
                    continue
                row = dict(zip(header, parts))

                label = row.get("Label", "").strip()
                if "Botnet" in label:
                    verdict = "ATTACK"
                elif "Normal" in label:
                    verdict = "FALSE POSITIVE"
                else:
                    continue

                if len(buckets[verdict]) >= CAP:
                    continue

                tot_pkts  = row.get("TotPkts",  "0").strip()
                src_bytes = row.get("SrcBytes",  "0").strip()
                tot_bytes = row.get("TotBytes",  "0").strip()
                try:
                    half      = str(int(float(tot_pkts)) // 2)
                    dst_bytes = str(max(0.0, float(tot_bytes) - float(src_bytes)))
                except ValueError:
                    half = "0"; dst_bytes = "0"

                buckets[verdict].append(make_sample(
                    proto      = row.get("Proto", "unknown").strip().lower(),
                    duration   = row.get("Dur", "0").strip(),
                    orig_pkts  = half,
                    resp_pkts  = half,
                    orig_bytes = src_bytes,
                    resp_bytes = dst_bytes,
                    conn_state = row.get("State", "-").strip(),
                    ground_truth = verdict,
                    source     = "ctu13",
                    raw_label  = label.strip(),
                    service    = "-",  # binetflow has no app-layer service field
                ))

    atk = len(buckets["ATTACK"])
    ben = len(buckets["FALSE POSITIVE"])
    print(f"  CTU-13: {atk} attacks, {ben} benign")
    return buckets["ATTACK"] + buckets["FALSE POSITIVE"]


def load_uwf(dataset_dir):
    """UWF-ZeekData24: real Zeek conn.log from UWF cyber range (MITRE-labeled).

    label_binary: "True" → ATTACK, "False" → FP
    label_tactic: MITRE tactic name (e.g. lateral_movement, command_and_control, none)
    """
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] UWF-ZeekData24 not found: {dataset_dir}")
        return []

    csv_files = sorted(glob.glob(os.path.join(dataset_dir, "**/*.csv"), recursive=True))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith(".")]
    if not csv_files:
        print(f"[SKIP] No CSVs in {dataset_dir}")
        return []

    print(f"[UWF-ZeekData24] {len(csv_files)} CSV(s) from {dataset_dir}")
    buckets = defaultdict(list)

    for fpath in csv_files:
        if all(len(buckets[v]) >= CAP for v in ["ATTACK", "FALSE POSITIVE"]):
            break
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"  ERROR {fpath}: {e}")
            continue
        df.columns = [c.strip() for c in df.columns]

        label_bin    = next((c for c in ["label_binary"] if c in df.columns), None)
        label_tactic = next((c for c in ["label_tactic"] if c in df.columns), None)
        if label_bin is None:
            continue

        for _, row in df.iterrows():
            verdict = ("ATTACK" if str(row[label_bin]).strip() == "True"
                       else "FALSE POSITIVE")
            if len(buckets[verdict]) >= CAP:
                continue

            # raw_label = MITRE tactic for attacks, "benign" for FP
            if label_tactic:
                raw = str(row[label_tactic]).strip()
                raw_label = raw if raw and raw != "none" else (
                    "Benign" if verdict == "FALSE POSITIVE" else "unknown_attack"
                )
            else:
                raw_label = "ATTACK" if verdict == "ATTACK" else "Benign"

            def _clean(val):
                s = str(val).strip()
                return "" if s in ("nan", "None", "NaN") else s

            svc = _clean(row.get("service", "-")) or "-"
            buckets[verdict].append(make_sample(
                proto      = _clean(row.get("proto", "unknown")),
                duration   = _clean(row.get("duration", "")),
                orig_pkts  = _clean(row.get("orig_pkts", "")),
                resp_pkts  = _clean(row.get("resp_pkts", "")),
                orig_bytes = _clean(row.get("orig_bytes", "")),
                resp_bytes = _clean(row.get("resp_bytes", "")),
                conn_state = _clean(row.get("conn_state", "-")) or "-",
                ground_truth = verdict,
                source     = "uwf",
                raw_label  = raw_label,
                service    = svc,
            ))

    atk = len(buckets["ATTACK"])
    ben = len(buckets["FALSE POSITIVE"])
    print(f"  UWF: {atk} attacks, {ben} benign")
    return buckets["ATTACK"] + buckets["FALSE POSITIVE"]


def load_ctu_normal(dataset_dir):
    """CTU-Normal: benign-only Zeek conn.log TSV (standard 21-field format)."""
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] CTU-Normal not found: {dataset_dir}")
        return []

    log_files = sorted(glob.glob(os.path.join(dataset_dir, "*.log")))
    if not log_files:
        print(f"[SKIP] No .log files in {dataset_dir}")
        return []

    print(f"[CTU-Normal] {len(log_files)} conn.log file(s) from {dataset_dir}")
    samples = []

    for fpath in log_files:
        if len(samples) >= CAP:
            break
        with open(fpath) as f:
            for line in f:
                if len(samples) >= CAP:
                    break
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 21:
                    continue
                samples.append(make_sample(
                    proto      = parts[6],
                    duration   = parts[8],
                    orig_pkts  = parts[16],
                    resp_pkts  = parts[18],
                    orig_bytes = parts[9],
                    resp_bytes = parts[10],
                    conn_state = parts[11],
                    ground_truth = "FALSE POSITIVE",
                    source     = "ctu_normal",
                    raw_label  = "Benign",
                    service    = parts[7],
                ))

    print(f"  CTU-Normal: 0 attacks, {len(samples)} benign")
    return samples


# ── Sample generation ───────────────────────────────────────────────────────────
def generate_samples():
    all_samples = []
    all_samples += load_iot23(DATASETS["iot23"])
    all_samples += load_ctu13(DATASETS["ctu13"])
    all_samples += load_uwf(DATASETS["uwf"])
    all_samples += load_ctu_normal(DATASETS["ctu_normal"])

    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)

    with open(CACHE_FILE, "w") as f:
        json.dump(all_samples, f, indent=2)

    atk = sum(1 for s in all_samples if s["ground_truth"] == "ATTACK")
    ben = sum(1 for s in all_samples if s["ground_truth"] == "FALSE POSITIVE")
    print(f"\n✅ {len(all_samples)} samples cached → {CACHE_FILE}")
    print(f"   Attacks: {atk}  |  Benign: {ben}")
    return all_samples


# ── Inference ───────────────────────────────────────────────────────────────────
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

    def __len__(self):           return len(self.texts)
    def __getitem__(self, i):    return self.texts[i]


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


# ── Reporting ───────────────────────────────────────────────────────────────────
SOURCE_NAMES = {
    "iot23":      "IoT-23        (Zeek conn.log)",
    "ctu13":      "CTU-13        (binetflow)",
    "uwf":        "UWF-ZeekData24(Zeek conn.log)",
    "ctu_normal": "CTU-Normal    (Zeek conn.log)",
}


def compute_metrics(preds, samples, unknowns):
    truths     = [s["ground_truth"] for s in samples]
    atk_idx    = [i for i, t in enumerate(truths) if t == "ATTACK"]
    ben_idx    = [i for i, t in enumerate(truths) if t == "FALSE POSITIVE"]
    mcc = matthews_corrcoef(
        [1 if t == "ATTACK" else 0 for t in truths],
        [1 if p == "ATTACK" else 0 for p in preds],
    )
    return {
        "accuracy":   sum(t == p for t, p in zip(truths, preds)) / len(truths),
        "atk_recall": sum(preds[i] == "ATTACK"         for i in atk_idx) / max(len(atk_idx), 1),
        "ben_recall": sum(preds[i] == "FALSE POSITIVE" for i in ben_idx) / max(len(ben_idx), 1),
        "fmt_fail":   unknowns / len(truths),
        "mcc":        mcc,
        "truths":     truths,
        "preds":      preds,
    }


def print_report(preds, samples, model_label, unknowns, out_lines):
    truths = [s["ground_truth"] for s in samples]
    labels = ["ATTACK", "FALSE POSITIVE"]
    mcc    = matthews_corrcoef(
        [1 if t == "ATTACK" else 0 for t in truths],
        [1 if p == "ATTACK" else 0 for p in preds],
    )
    n   = len(samples)
    atk = sum(1 for t in truths if t == "ATTACK")
    ben = n - atk

    lines = [
        f"\n{'='*70}",
        f"  MODEL  : {model_label}",
        f"  Samples: {n}  (attacks: {atk}, benign: {ben})",
        f"  Format failures : {unknowns} ({100*unknowns/n:.1f}%)",
        f"  MCC             : {mcc:+.4f}  (-1 to +1, higher is better)",
        f"{'='*70}",
        classification_report(truths, preds, labels=labels, zero_division=0),
        "Confusion Matrix  (rows = actual, cols = predicted)",
        f"{'':22s} {'ATTACK':>10} {'FALSE POSITIVE':>15}",
    ]
    cm = confusion_matrix(truths, preds, labels=labels)
    for row_label, row in zip(labels, cm):
        pct = [f"{100*v/max(row.sum(),1):.0f}%" for v in row]
        lines.append(f"  {row_label:20s} {row[0]:>6} ({pct[0]:>4})  {row[1]:>6} ({pct[1]:>4})")

    # Per-source breakdown
    lines.append(f"\n--- Per source ---")
    lines.append(f"  {'Source':<34} {'Atk':>5} {'Recall':>7}   {'Ben':>5} {'Recall':>7}   {'Acc':>6}")
    lines.append(f"  {'-'*34} {'-'*5} {'-'*7}   {'-'*5} {'-'*7}   {'-'*6}")
    for src in sorted(set(s["source"] for s in samples)):
        idx   = [i for i, s in enumerate(samples) if s["source"] == src]
        t_sub = [truths[i] for i in idx]
        p_sub = [preds[i]  for i in idx]
        a_idx = [i for i, t in enumerate(t_sub) if t == "ATTACK"]
        b_idx = [i for i, t in enumerate(t_sub) if t == "FALSE POSITIVE"]
        a_rec = sum(p_sub[i] == "ATTACK"         for i in a_idx) / max(len(a_idx), 1)
        b_rec = sum(p_sub[i] == "FALSE POSITIVE" for i in b_idx) / max(len(b_idx), 1)
        acc   = sum(t == p for t, p in zip(t_sub, p_sub)) / len(t_sub)
        name  = SOURCE_NAMES.get(src, src)
        lines.append(
            f"  {name:<34} {len(a_idx):>5} {a_rec:>7.1%}   "
            f"{len(b_idx):>5} {b_rec:>7.1%}   {acc:>6.1%}"
        )

    # Per-label breakdown (attack types + benign sub-labels)
    lines.append(f"\n--- Per label (attacks only) ---")
    atk_labels = sorted(set(
        s["raw_label"] for s in samples if s["ground_truth"] == "ATTACK"
    ))
    for rl in atk_labels:
        idx     = [i for i, s in enumerate(samples)
                   if s["raw_label"] == rl and s["ground_truth"] == "ATTACK"]
        correct = sum(preds[i] == "ATTACK" for i in idx)
        lines.append(f"  {rl:44s} {correct:>3}/{len(idx):<3} ({100*correct/len(idx):>3.0f}%)")

    for line in lines:
        print(line)
    out_lines.extend(lines)


# ── Model loading ───────────────────────────────────────────────────────────────
def load_model(adapter_path):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)
    print(f"\nLoading {adapter_path} ...")
    base  = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                                  device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    regen = "--regen" in sys.argv

    if not regen and os.path.exists(CACHE_FILE):
        print(f"[CACHE] Loading {CACHE_FILE}")
        with open(CACHE_FILE) as f:
            samples = json.load(f)
        print(f"  {len(samples)} samples loaded")
    else:
        samples = generate_samples()

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    ts        = datetime.now().strftime("%Y-%m-%d %H:%M")
    atk_n     = sum(1 for s in samples if s["ground_truth"] == "ATTACK")
    ben_n     = len(samples) - atk_n
    out_lines = [
        f"REAL-WORLD ZEEK BENCHMARK — {ts}",
        f"Sources: IoT-23, CTU-13, UWF-ZeekData24, CTU-Normal",
        f"Samples: {len(samples)} total ({atk_n} attacks / {ben_n} benign) | Seed: {RANDOM_SEED}",
    ]
    results     = []
    json_output = {
        "timestamp": datetime.now().isoformat(),
        "samples":   len(samples),
        "sources":   {src: sum(1 for s in samples if s["source"] == src)
                      for src in ["iot23", "ctu13", "uwf", "ctu_normal"]},
        "models":    [],
    }

    for label, adapter_path in MODELS:
        if not os.path.isdir(adapter_path):
            print(f"\n[SKIP] {label} — not found at {adapter_path}")
            continue

        model           = load_model(adapter_path)
        preds, unknowns = run_inference(model, samples, label)
        print_report(preds, samples, label, unknowns, out_lines)
        m               = compute_metrics(preds, samples, unknowns)
        results.append((label, m))

        # Per-source metrics for JSON
        src_metrics = {}
        for src in ["iot23", "ctu13", "uwf", "ctu_normal"]:
            idx   = [i for i, s in enumerate(samples) if s["source"] == src]
            if not idx:
                continue
            t_sub = [m["truths"][i] for i in idx]
            p_sub = [m["preds"][i]  for i in idx]
            a_idx = [i for i, t in enumerate(t_sub) if t == "ATTACK"]
            b_idx = [i for i, t in enumerate(t_sub) if t == "FALSE POSITIVE"]
            src_metrics[src] = {
                "n":          len(idx),
                "accuracy":   round(sum(t == p for t, p in zip(t_sub, p_sub)) / len(t_sub), 4),
                "atk_recall": round(sum(p_sub[i] == "ATTACK"         for i in a_idx) / max(len(a_idx), 1), 4),
                "ben_recall": round(sum(p_sub[i] == "FALSE POSITIVE" for i in b_idx) / max(len(b_idx), 1), 4),
            }

        json_output["models"].append({
            "label":      label,
            "accuracy":   round(m["accuracy"],   4),
            "atk_recall": round(m["atk_recall"], 4),
            "ben_recall": round(m["ben_recall"], 4),
            "fmt_fail":   round(m["fmt_fail"],   4),
            "mcc":        round(m["mcc"],        4),
            "per_source": src_metrics,
        })

        del model
        torch.cuda.empty_cache()

    # ── Comparison table ─────────────────────────────────────────────────────────
    if len(results) >= 2:
        labels_str = " vs ".join(lbl for lbl, _ in results)
        summary = [
            f"\n{'='*74}",
            f"  REAL-WORLD ZEEK COMPARISON — {labels_str}",
            f"{'='*74}",
            f"  {'Model':<20} {'Accuracy':>9} {'Atk Recall':>11} {'FP Recall':>10} {'MCC':>7} {'Fmt Fail':>9}",
            f"  {'-'*20} {'-'*9} {'-'*11} {'-'*10} {'-'*7} {'-'*9}",
        ]
        for lbl, m in results:
            summary.append(
                f"  {lbl:<20} {m['accuracy']:>9.1%} {m['atk_recall']:>11.1%}"
                f" {m['ben_recall']:>10.1%} {m['mcc']:>+7.3f} {m['fmt_fail']:>9.1%}"
            )

        # Delta rows: each version vs the previous
        summary.append(f"  {'-'*20} {'-'*9} {'-'*11} {'-'*10} {'-'*7} {'-'*9}")
        for i in range(1, len(results)):
            l_prev, m_prev = results[i - 1]
            l_cur,  m_cur  = results[i]
            short_prev = l_prev.split()[0]  # e.g. "v4"
            short_cur  = l_cur.split()[0]   # e.g. "v6"
            delta_label = f"delta ({short_cur} - {short_prev})"
            summary.append(
                f"  {delta_label:<20}"
                f" {m_cur['accuracy']   - m_prev['accuracy']:>+9.1%}"
                f" {m_cur['atk_recall'] - m_prev['atk_recall']:>+11.1%}"
                f" {m_cur['ben_recall'] - m_prev['ben_recall']:>+10.1%}"
                f" {m_cur['mcc']        - m_prev['mcc']:>+7.3f}"
                f" {m_cur['fmt_fail']   - m_prev['fmt_fail']:>+9.1%}"
            )

        summary += [
            f"{'='*74}",
            "",
            "  Atk Recall = % of actual attacks caught (sensitivity / TPR)",
            "  FP Recall  = % of benign flows correctly identified (specificity)",
            "  MCC        = Matthews Correlation Coefficient (-1 to +1)",
            "  Fmt Fail   = % of outputs missing VERDICT line",
            "",
            "  Sources: IoT-23 + CTU-13 + UWF-ZeekData24 + CTU-Normal",
            "  (native Zeek conn.log — NO synthetic field mapping)",
        ]

        # Per-source delta table (last model vs first)
        if (len(json_output["models"]) >= 2 and
                "per_source" in json_output["models"][0] and
                "per_source" in json_output["models"][-1]):
            ps_first = json_output["models"][0]["per_source"]
            ps_last  = json_output["models"][-1]["per_source"]
            l_first  = json_output["models"][0]["label"].split()[0]
            l_last   = json_output["models"][-1]["label"].split()[0]
            summary += [
                f"\n--- Per-source delta ({l_last} - {l_first}) ---",
                f"  {'Source':<34} {'Δ Accuracy':>11} {'Δ Atk Recall':>13} {'Δ FP Recall':>12}",
                f"  {'-'*34} {'-'*11} {'-'*13} {'-'*12}",
            ]
            for src in ["iot23", "ctu13", "uwf", "ctu_normal"]:
                if src not in ps_first or src not in ps_last:
                    continue
                da  = ps_last[src]["accuracy"]   - ps_first[src]["accuracy"]
                dar = ps_last[src]["atk_recall"] - ps_first[src]["atk_recall"]
                dbr = ps_last[src]["ben_recall"] - ps_first[src]["ben_recall"]
                name = SOURCE_NAMES.get(src, src)
                summary.append(
                    f"  {name:<34} {da:>+11.1%} {dar:>+13.1%} {dbr:>+12.1%}"
                )

        for line in summary:
            print(line)
        out_lines.extend(summary)

    with open(REPORT_TXT, "w") as f:
        f.write("\n".join(out_lines))
    with open(RESULTS_JSON, "w") as f:
        json.dump(json_output, f, indent=2)

    print(f"\n✅ Report  → {REPORT_TXT}")
    print(f"✅ Results → {RESULTS_JSON}")
