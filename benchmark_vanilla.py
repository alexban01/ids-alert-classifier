"""Standalone vanilla Qwen2.5-1.5B benchmark on CICIDS2017 (Zeek-native prompt)."""

import os
import json
import random
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ───────────────────────────────────────────────────────────────────
BASE_MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
BENCHMARK_CACHE = "benchmark_samples_v4.json"
MAX_NEW_TOKENS  = 80
BATCH_SIZE      = 8
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

CICIDS_COLS = [
    "Protocol", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Label",
]

SYSTEM_PROMPT = (
    "You are a network security analyst. "
    "Always respond with VERDICT: <ATTACK or FALSE POSITIVE> on the first line, "
    "followed by REASON: <brief explanation>."
)

# ── Zeek-native prompt builder (mirrors preprocess_zeek.py) ──────────────────
def _safe(v, fmt=".1f"):
    try:
        return format(float(v), fmt) if v not in (None, "", "-", "?") else "N/A"
    except (ValueError, TypeError):
        return "N/A"

def build_prompt(proto, duration, orig_pkts, resp_pkts,
                 orig_bytes, resp_bytes, conn_state):
    try:
        dur_f = float(duration)
        ob_f  = float(orig_bytes)
        rb_f  = float(resp_bytes)
        op_f  = float(orig_pkts)
        rp_f  = float(resp_pkts)
        bps   = (ob_f + rb_f) / dur_f if dur_f > 0 else 0.0
        op_sz = ob_f / op_f if op_f > 0 else 0.0
        rp_sz = rb_f / rp_f if rp_f > 0 else 0.0
    except (ValueError, TypeError, ZeroDivisionError):
        bps = op_sz = rp_sz = 0.0

    lines = [
        "Analyze this network connection and classify it as ATTACK or FALSE POSITIVE.\n",
        f"  Proto:              {proto}",
        f"  Duration (s):       {_safe(duration, '.6f')}",
        f"  Orig Packets:       {_safe(orig_pkts, '.0f')}",
        f"  Resp Packets:       {_safe(resp_pkts, '.0f')}",
        f"  Orig Bytes:         {_safe(orig_bytes, '.0f')}",
        f"  Resp Bytes:         {_safe(resp_bytes, '.0f')}",
        f"  Conn State:         {conn_state}",
        f"  Bytes/sec:          {_safe(bps, '.1f')}",
        f"  Orig Bytes/Pkt:     {_safe(op_sz, '.1f')}",
        f"  Resp Bytes/Pkt:     {_safe(rp_sz, '.1f')}",
    ]
    return "\n".join(lines)

def label_to_verdict(label):
    return "FALSE POSITIVE" if label.strip() == "BENIGN" else "ATTACK"

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

        avail = [c for c in CICIDS_COLS if c in df.columns]
        if "Label" not in df.columns:
            print(f"  [SKIP] No Label column")
            continue

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

            orig_pkts  = str(row["Total Fwd Packets"])          if "Total Fwd Packets"          in avail else "0"
            resp_pkts  = str(row["Total Backward Packets"])     if "Total Backward Packets"     in avail else "0"
            orig_bytes = str(row["Total Length of Fwd Packets"])if "Total Length of Fwd Packets"in avail else "0"
            resp_bytes = str(row["Total Length of Bwd Packets"])if "Total Length of Bwd Packets"in avail else "0"
            conn_state = "-"

            prompt = build_prompt(proto_num, duration, orig_pkts, resp_pkts,
                                  orig_bytes, resp_bytes, conn_state)

            samples.append({
                "prompt":       prompt,
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

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    if os.path.exists(BENCHMARK_CACHE):
        print(f"[CACHE] Loading existing samples from {BENCHMARK_CACHE}")
        with open(BENCHMARK_CACHE) as f:
            samples = json.load(f)
        print(f"  {len(samples)} samples loaded")
    else:
        samples = generate_benchmark_samples()

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    print(f"Loading vanilla model: {BASE_MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="cuda",
    )
    model.eval()

    preds, unknowns = run_batched_inference(
        model, samples, "Vanilla Qwen2.5-1.5B-Instruct"
    )
    print_report(preds, samples, "Vanilla Qwen2.5-1.5B-Instruct", unknowns)
