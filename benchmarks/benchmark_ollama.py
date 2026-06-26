"""
benchmark_ollama.py — Benchmark a model served by Ollama.

Reuses cached sample files from benchmark_realworld.py or benchmark_v6.py.
No GPU/transformers required — calls the Ollama HTTP API.

Usage:
    .venv/bin/python benchmark_ollama.py [MODEL] [--cache FILE]

    MODEL        Ollama model name (default: ids-classifier)
    --cache FILE Sample cache JSON (default: results/benchmark_realworld_cache.json)

Examples:
    .venv/bin/python benchmark_ollama.py
    .venv/bin/python benchmark_ollama.py ids-classifier
    .venv/bin/python benchmark_ollama.py ids-classifier --cache results/benchmark_samples_v4.json
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

from ids.prompt_utils import SYSTEM_PROMPT, extract_verdict

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "ids-classifier"
DEFAULT_CACHE = "results/benchmark_realworld_cache.json"

SOURCE_NAMES = {
    "iot23":      "IoT-23        (Zeek conn.log)",
    "ctu13":      "CTU-13        (binetflow)",
    "uwf":        "UWF-ZeekData24(Zeek conn.log)",
    "ctu_normal": "CTU-Normal    (Zeek conn.log)",
    # CICIDS2017 cache uses "source_file" not "source", handled below
}

# ── Ollama call ─────────────────────────────────────────────────────────────────
def build_qwen_prompt(system, user):
    """Format prompt using Qwen2.5 chat template manually.

    Bypasses Ollama's template handling — works even if the GGUF has no
    embedded chat_template metadata (common after llama.cpp GGUF conversion).
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def ollama_classify(model, prompt, timeout=30):
    """POST to Ollama /api/generate with raw Qwen2.5 prompt. Returns response text."""
    payload = json.dumps({
        "model":  model,
        "prompt": build_qwen_prompt(SYSTEM_PROMPT, prompt),
        "stream": False,
        "raw":    True,   # skip Ollama's template — we already applied it
    }).encode()

    req = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["response"]
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable at {OLLAMA_GENERATE_URL}: {e}") from e


def check_ollama(model):
    """Verify Ollama is up and the model is available."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
        names = [m["name"] for m in tags.get("models", [])]
        # Ollama sometimes stores name as "model:latest"
        available = any(n == model or n.startswith(model + ":") for n in names)
        if not available:
            print(f"[WARN] Model '{model}' not found in Ollama. Available: {names}")
            print(f"       Run:  ollama create {model} -f Modelfile")
    except urllib.error.URLError as e:
        print(f"[ERROR] Ollama not running: {e}")
        print(f"        Start it with:  ollama serve")
        sys.exit(1)


# ── Reporting ───────────────────────────────────────────────────────────────────
def print_report(preds, samples, model_label, unknowns, elapsed, out_lines):
    truths = [s["ground_truth"] for s in samples]
    labels = ["ATTACK", "FALSE POSITIVE"]
    n      = len(samples)
    atk    = sum(1 for t in truths if t == "ATTACK")
    ben    = n - atk
    mcc    = matthews_corrcoef(
        [1 if t == "ATTACK" else 0 for t in truths],
        [1 if p == "ATTACK" else 0 for p in preds],
    )

    lines = [
        f"\n{'='*70}",
        f"  MODEL  : {model_label}",
        f"  Samples: {n}  (attacks: {atk}, benign: {ben})",
        f"  Elapsed: {elapsed:.1f}s  ({elapsed/n:.2f}s/sample)",
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

    # Determine which source field the cache uses
    src_field = "source" if "source" in samples[0] else "source_file"

    # Per-source breakdown
    sources = sorted(set(s[src_field] for s in samples))
    if len(sources) > 1:
        lines.append(f"\n--- Per source ---")
        lines.append(f"  {'Source':<38} {'Atk':>5} {'Recall':>7}   {'Ben':>5} {'Recall':>7}   {'Acc':>6}")
        lines.append(f"  {'-'*38} {'-'*5} {'-'*7}   {'-'*5} {'-'*7}   {'-'*6}")
        for src in sources:
            idx   = [i for i, s in enumerate(samples) if s[src_field] == src]
            t_sub = [truths[i] for i in idx]
            p_sub = [preds[i]  for i in idx]
            a_idx = [i for i, t in enumerate(t_sub) if t == "ATTACK"]
            b_idx = [i for i, t in enumerate(t_sub) if t == "FALSE POSITIVE"]
            a_rec = sum(p_sub[i] == "ATTACK"         for i in a_idx) / max(len(a_idx), 1)
            b_rec = sum(p_sub[i] == "FALSE POSITIVE" for i in b_idx) / max(len(b_idx), 1)
            acc   = sum(t == p for t, p in zip(t_sub, p_sub)) / len(t_sub)
            name  = SOURCE_NAMES.get(src, src)
            lines.append(
                f"  {name:<38} {len(a_idx):>5} {a_rec:>7.1%}   "
                f"{len(b_idx):>5} {b_rec:>7.1%}   {acc:>6.1%}"
            )

    # Per-label breakdown (attack types only)
    rl_field = "raw_label" if "raw_label" in samples[0] else "raw_label"
    if rl_field in samples[0]:
        atk_labels = sorted(set(
            s[rl_field] for s in samples if s["ground_truth"] == "ATTACK"
        ))
        if atk_labels:
            lines.append(f"\n--- Per label (attacks only) ---")
            for rl in atk_labels:
                idx     = [i for i, s in enumerate(samples)
                           if s.get(rl_field) == rl and s["ground_truth"] == "ATTACK"]
                correct = sum(preds[i] == "ATTACK" for i in idx)
                lines.append(
                    f"  {rl:44s} {correct:>3}/{len(idx):<3} ({100*correct/max(len(idx),1):>3.0f}%)"
                )

    for line in lines:
        print(line)
    out_lines.extend(lines)

    return mcc


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Parse args
    args       = sys.argv[1:]
    model_name = DEFAULT_MODEL
    cache_file = DEFAULT_CACHE
    i = 0
    while i < len(args):
        if args[i] == "--cache" and i + 1 < len(args):
            cache_file = args[i + 1]; i += 2
        elif not args[i].startswith("--"):
            model_name = args[i]; i += 1
        else:
            i += 1

    check_ollama(model_name)

    print(f"[CACHE] Loading {cache_file}")
    with open(cache_file) as f:
        samples = json.load(f)
    print(f"  {len(samples)} samples loaded")

    atk_n = sum(1 for s in samples if s["ground_truth"] == "ATTACK")
    ben_n = len(samples) - atk_n
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")

    out_lines = [
        f"OLLAMA BENCHMARK — {ts}",
        f"Model  : {model_name}",
        f"Cache  : {cache_file}",
        f"Samples: {len(samples)} ({atk_n} attacks / {ben_n} benign)",
    ]

    preds    = []
    unknowns = 0
    t0       = time.time()

    print(f"\nRunning inference: {model_name}")
    for i, s in enumerate(samples):
        try:
            text    = ollama_classify(model_name, s["prompt"])
            verdict = extract_verdict(text)
        except RuntimeError as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)

        if verdict == "UNKNOWN":
            unknowns += 1
        preds.append(verdict)

        done = i + 1
        if done % 50 == 0 or done == len(samples):
            elapsed = time.time() - t0
            rate    = done / elapsed
            eta     = (len(samples) - done) / rate if rate > 0 else 0
            print(f"  {done}/{len(samples)}  {rate:.1f} samples/s  ETA {eta:.0f}s  ...", end="\r")

    elapsed = time.time() - t0
    print()

    mcc = print_report(preds, samples, model_name, unknowns, elapsed, out_lines)

    report_file  = f"results/benchmark_ollama_{model_name.replace(':', '_')}_report.txt"
    results_file = f"results/benchmark_ollama_{model_name.replace(':', '_')}_results.json"

    truths = [s["ground_truth"] for s in samples]
    accuracy   = sum(t == p for t, p in zip(truths, preds)) / len(truths)
    atk_idx    = [i for i, t in enumerate(truths) if t == "ATTACK"]
    ben_idx    = [i for i, t in enumerate(truths) if t == "FALSE POSITIVE"]
    atk_recall = sum(preds[i] == "ATTACK"         for i in atk_idx) / max(len(atk_idx), 1)
    ben_recall = sum(preds[i] == "FALSE POSITIVE" for i in ben_idx) / max(len(ben_idx), 1)

    results = {
        "timestamp":  datetime.now().isoformat(),
        "model":      model_name,
        "cache":      cache_file,
        "samples":    len(samples),
        "elapsed_s":  round(elapsed, 1),
        "accuracy":   round(accuracy,   4),
        "atk_recall": round(atk_recall, 4),
        "ben_recall": round(ben_recall, 4),
        "fmt_fail":   round(unknowns / len(samples), 4),
        "mcc":        round(mcc, 4),
    }

    with open(report_file, "w") as f:
        f.write("\n".join(out_lines))
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Report  → {report_file}")
    print(f"✅ Results → {results_file}")
