"""
benchmark_realworld.py — real-world Zeek benchmark for IDS classifier.

Uses native Zeek conn.log sources (IoT-23, CTU-13, UWF-ZeekData24, CTU-Normal)
with authentic conn_state values and protocol names — unlike benchmark_v6.py
which uses CICIDS2017 (CICFlowMeter CSVs with synthetic field mapping).

Usage:
    .venv/bin/python benchmark_realworld.py [--regen] [--ood] [--no-behavior]

    --regen         Force regeneration of the entire sample cache.
    --ood           Run inference only on OOD (CTU-SME-11 Windows7AD-1) samples.
    --regen --ood   Regenerate only the OOD samples in the cache, then run
                    OOD-only inference (does not touch other source caches).
    --no-behavior   Keep prompts conn-only (skip [BEHAVIOR] rebuild).
    --host-pass2    Run the host-level aggregation pass (OFF by default — it is a
                    documented null result; see thesis_notes_12.txt).
"""

import os
import sys
import json
import random
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

from bench_loaders import (
    CAP, load_iot23, load_ctu13, load_uwf, load_ctu_normal,
    load_ctu_sme11, load_ctu_win7ad, load_ctu_botnet3,
)
from ids.behavior_features import build_behavior_contexts, build_host_summaries
from ids.infer_utils import (BASE_MODEL, chat_text, load_lora_model, load_tokenizer,
                             resolve_system_prompt)
from ids.prompt_utils import (build_prompt, build_host_prompt, extract_verdict, extract_reason,
                              SYSTEM_PROMPT)

# ── Config ──────────────────────────────────────────────────────────────────────
CACHE_FILE     = "results/benchmark_realworld_cache.json"
REPORT_TXT     = "results/benchmark_realworld_report.txt"
RESULTS_JSON   = "results/benchmark_realworld_results.json"
MAX_NEW_TOKENS = 80
BATCH_SIZE     = 24
RANDOM_SEED    = 42

MODELS = [
    # ("v4 Fine-tuned",        "./models/v4-ids-lora-adapter"),
    # ("v6 Fine-tuned",        "./models/v6-ids-lora-adapter"),
    # ("v7.1 Fine-tuned",      "./models/v7.1-ids-lora-adapter"),
    # ("v8 ckpt-1500 (ep1)",   "./models/v8-ids-model/checkpoint-1500"),
    # ("v8.1 Fine-tuned",      "./models/v8.1-ids-lora-adapter"),
    # ("v9.0 ckpt-1420 (ep1)",   "./models/v9.0-ids-model/checkpoint-1420"),
    # ("v9.1 ckpt-1186 (ep1)",   "./models/v9.1-ids-model/checkpoint-1186"),
    # ("v9.1 Fine-tuned",        "./models/v9.1-ids-lora-adapter"),   # superseded by v11
    # ("v11 ckpt-11313 (ep1)",   "./models/v11-ids-model/checkpoint-11313"),
    # ("v11 ckpt-22626 (ep2)",   "./models/v11-ids-model/checkpoint-22626"),   # best (eval_loss 0.1570)
    # v12 / v12.1: no-reason (verdict-only) runs, both interrupted just past
    # epoch 1 — ckpt-10000 ≈ epoch-1 boundary. run.json reconstructed post-hoc.
    # ("v12 ckpt-10000 (ep1)",      "./models/v12-ids-model/checkpoint-10000"),
    # ("v12.1 ckpt-10000 (ep1)",    "./models/v12.1-ids-model/checkpoint-10000"),
    # v12.2: 50% downsampled data, 1 epoch, complete run. CAVEAT: trained at
    # r=16/alpha=32 (train.py had already flipped to the v13 setting), NOT
    # v12.1's r=32/64 — so it's "v13 on half data", not a pure volume ablation.
    # ("v12.2 adapter (ep1)",       "./models/v12.2-ids-lora-adapter"),
    ("v13.1 adapter (ep1)",       "./models/v13.1-ids-lora-adapter"),
]

DATASETS = {
    "iot23":      "datasets/iot-23/iot_23_datasets_small.tar.gz",
    "ctu13":      "datasets/ctu-13/CTU-13-Dataset/",
    "uwf":        "datasets/uwf-zeekdata24/",
    "ctu_normal": "datasets/ctu-normal/",
}

SOURCE_NAMES = {
    "iot23":        "IoT-23          (Zeek conn.log)",
    "ctu13":        "CTU-13          (binetflow)",
    "uwf":          "UWF-ZeekData24  (Zeek conn.log)",
    "ctu_normal":   "CTU-Normal      (Zeek conn.log)",
    "ctu_win7ad":   "CTU-SME-11 [OOD-Hard] (Win7AD-1)",
    "ctu_sme11":    "CTU-SME-11 [OOD-Easy] (Echo)",
    "ctu_botnet3":  "CTU-Malware3   [OOD-Floor] (Kelihos)",
}

# OOD sources — held out from training, used for out-of-distribution evaluation.
OOD_SOURCES  = {"ctu_win7ad", "ctu_sme11", "ctu_botnet3"}
ALL_SOURCES  = ["iot23", "ctu13", "uwf", "ctu_normal", "ctu_win7ad", "ctu_sme11", "ctu_botnet3"]

# ── Sample generation ────────────────────────────────────────────────────────────
_LOADER_ORDER = ["iot23", "ctu13", "uwf", "ctu_normal", "ctu_win7ad", "ctu_sme11", "ctu_botnet3"]


def _run_bench_loader(job_name):
    """Worker wrapper for ProcessPoolExecutor loader jobs."""
    if job_name == "iot23":
        return job_name, load_iot23(DATASETS["iot23"])
    if job_name == "ctu13":
        return job_name, load_ctu13(DATASETS["ctu13"])
    if job_name == "uwf":
        return job_name, load_uwf(DATASETS["uwf"])
    if job_name == "ctu_normal":
        return job_name, load_ctu_normal(DATASETS["ctu_normal"])
    if job_name == "ctu_win7ad":
        return job_name, load_ctu_win7ad()
    if job_name == "ctu_sme11":
        return job_name, load_ctu_sme11()
    if job_name == "ctu_botnet3":
        return job_name, load_ctu_botnet3()
    raise ValueError(f"Unknown loader job: {job_name}")


def generate_samples():
    """Run all loaders in parallel; collect in fixed order for deterministic shuffle."""
    max_workers = min(len(_LOADER_ORDER), os.cpu_count() or 1)
    all_samples = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {name: executor.submit(_run_bench_loader, name)
                   for name in _LOADER_ORDER}
        for name in _LOADER_ORDER:
            try:
                _, job_samples = futures[name].result()
            except Exception as e:
                print(f"[ERROR] Loader '{name}' failed: {e}")
                job_samples = []
            all_samples += job_samples
            print(f"[DONE] Loader '{name}': {len(job_samples)} samples")

    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)
    _save_cache(all_samples)
    return all_samples


def regen_ood_samples():
    """Replace only OOD samples in the cache; leave other sources untouched."""
    if not os.path.exists(CACHE_FILE):
        print("[REGEN-OOD] No cache found — running full generation instead.")
        return generate_samples()

    print(f"[REGEN-OOD] Loading existing cache: {CACHE_FILE}")
    with open(CACHE_FILE) as f:
        samples = json.load(f)

    non_ood = [s for s in samples if s["source"] not in OOD_SOURCES]
    old_ood_n = len(samples) - len(non_ood)
    print(f"  Dropping {old_ood_n} cached OOD samples (win7ad + sme11 + botnet3)")

    ood_samples = load_ctu_win7ad() + load_ctu_sme11() + load_ctu_botnet3()
    merged = non_ood + ood_samples
    random.seed(RANDOM_SEED)
    random.shuffle(merged)
    _save_cache(merged)
    print(f"  Replaced with {len(ood_samples)} fresh OOD samples")
    return merged


def _save_cache(samples):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(samples, f, indent=2)
    atk = sum(1 for s in samples if s["ground_truth"] == "ATTACK")
    ben = sum(1 for s in samples if s["ground_truth"] == "FALSE POSITIVE")
    print(f"\n✅ {len(samples)} samples cached → {CACHE_FILE}")
    print(f"   Attacks: {atk}  |  Benign: {ben}")


# ── Behavior / host passes ───────────────────────────────────────────────────────

def rebuild_prompts_with_behavior(samples):
    """Rebuild sample prompts with [BEHAVIOR] sections when raw fields allow it."""
    grouped = defaultdict(list)
    for idx, sample in enumerate(samples):
        grouped[(sample.get("source"), sample.get("group_id", sample.get("source")))].append((idx, sample))

    behavior_ctxs = [None] * len(samples)
    for items in grouped.values():
        rows = []
        order = []
        for idx, sample in items:
            if sample.get("ts") in (None, "", "-", "?", "None"):
                continue
            rows.append({
                "ts":         sample.get("ts"),
                "uid":        sample.get("uid"),
                "orig_h":     sample.get("orig_h"),
                "orig_p":     sample.get("orig_p"),
                "resp_h":     sample.get("resp_h"),
                "resp_p":     sample.get("resp_p"),
                "proto":      sample.get("proto"),
                "service":    sample.get("service"),
                "duration":   sample.get("duration"),
                "orig_bytes": sample.get("orig_bytes"),
                "resp_bytes": sample.get("resp_bytes"),
                "conn_state": sample.get("conn_state"),
                "orig_pkts":  sample.get("orig_pkts"),
                "resp_pkts":  sample.get("resp_pkts"),
            })
            order.append(idx)
        ctxs = build_behavior_contexts(rows)
        for idx, ctx in zip(order, ctxs):
            behavior_ctxs[idx] = ctx

    for i, sample in enumerate(samples):
        sample["behavior_ctx"] = behavior_ctxs[i]
        sample["prompt"] = build_prompt(
            sample.get("proto"), sample.get("duration"), sample.get("orig_pkts"), sample.get("resp_pkts"),
            sample.get("orig_bytes"), sample.get("resp_bytes"), sample.get("conn_state"), sample.get("service", "-"),
            resp_port=sample.get("resp_p", "-"), orig_port=sample.get("orig_p", "-"),
            behavior_ctx=behavior_ctxs[i],
        )
    return samples


def build_host_benchmark_samples(samples, preds):
    """Aggregate flow samples/predictions into host-level benchmark items."""
    rows = []
    flow_results = []
    for sample, pred in zip(samples, preds):
        rows.append({
            "host_key": f"{sample.get('source')}|{sample.get('group_id', sample.get('source'))}|{sample.get('orig_h') or '?'}",
            "orig_h": sample.get("orig_h") or "?",
            "uid": sample.get("uid"),
            "resp_h": sample.get("resp_h"),
            "resp_p": sample.get("resp_p"),
            "service": sample.get("service"),
            "conn_state": sample.get("conn_state"),
        })
        flow_results.append((pred, ""))

    host_summaries = build_host_summaries(rows, flow_results)
    host_truth = {}
    for sample in samples:
        key = f"{sample.get('source')}|{sample.get('group_id', sample.get('source'))}|{sample.get('orig_h') or '?'}"
        cur = host_truth.get(key, "FALSE POSITIVE")
        if sample.get("ground_truth") == "ATTACK":
            cur = "ATTACK"
        host_truth[key] = cur

    host_samples = []
    for summary in host_summaries:
        key = summary.get("host_key")
        if key not in host_truth:
            continue
        host_samples.append({
            "prompt": build_host_prompt(summary["host"], summary),
            "ground_truth": host_truth[key],
            "source": key.split("|", 1)[0],
            "raw_label": "HostAttack" if host_truth[key] == "ATTACK" else "HostBenign",
            "host": summary["host"],
        })
    return host_samples


# ── Inference ────────────────────────────────────────────────────────────────────
_tokenizer = None


class PromptDataset(Dataset):
    def __init__(self, samples, system_prompt=SYSTEM_PROMPT):
        self.texts = [chat_text(_tokenizer, s["prompt"], system_prompt) for s in samples]

    def __len__(self):        return len(self.texts)
    def __getitem__(self, i): return self.texts[i]


def collate_fn(batch):
    return _tokenizer(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512)


def run_inference(model, samples, label, system_prompt=SYSTEM_PROMPT):
    loader   = DataLoader(PromptDataset(samples, system_prompt), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True,
                          collate_fn=collate_fn)
    preds    = []
    reasons  = []
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
                reasons.append(extract_reason(text))
            print(f"  {min((i+1)*BATCH_SIZE, len(samples))}/{len(samples)}...", end="\r")
    print()
    return preds, unknowns, reasons


# ── Model loading ─────────────────────────────────────────────────────────────────
def load_model(adapter_path):
    print(f"\nLoading {adapter_path} ...")
    return load_lora_model(adapter_path)


def provenance_lines(label, adapter_path, run_info, system_prompt):
    """Banner: what the model was trained on (from run.json) + which prompt we serve it.

    The REASON line is the crux — a --no-reason model (dataset.reason == False) is
    auto-served the verdict-only prompt so train/inference formats match.
    """
    prompt_kind = "VERDICT+REASON" if "REASON" in system_prompt else "verdict-only"
    lines = [f"\n── Training provenance: {label} ──"]
    if run_info is None:
        lines += [
            f"  run.json    : none found in {adapter_path}",
            f"  REASON      : unknown → serving default ({prompt_kind}) system prompt",
            f"  (Adapter predates run manifests; retrain via train.py to record provenance.)",
        ]
        return lines
    hp     = run_info.get("hyperparams") or {}
    ds     = run_info.get("dataset")     or {}
    reason = {True: "ON", False: "OFF", None: "unknown"}[ds.get("reason")]
    match  = ds.get("matches_meta")
    lines += [
        f"  Trained     : {(run_info.get('created') or '?')[:10]}  git {run_info.get('git_sha') or '?'}"
        f"  on {run_info.get('target') or '?'}",
        f"  REASON      : {reason}  → serving {prompt_kind} system prompt",
        f"  Settings    : epochs={hp.get('epochs')}  packing={hp.get('packing')}  "
        f"TF={ds.get('training_factor')}  lora_r={hp.get('lora_r')}  eff_batch={hp.get('effective_batch')}",
        f"  Eval loss   : {run_info.get('eval_loss')}",
        f"  Dataset hash: {(ds.get('train_sha256') or '?')[:12]}"
        + ("" if match is None else f"  (matches local dataset meta: {match})"),
    ]
    return lines


# ── Reporting ─────────────────────────────────────────────────────────────────────
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


def print_comparison_table(results, json_output, out_lines):
    if len(results) < 2:
        return
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

    summary.append(f"  {'-'*20} {'-'*9} {'-'*11} {'-'*10} {'-'*7} {'-'*9}")
    for i in range(1, len(results)):
        l_prev, m_prev = results[i - 1]
        l_cur,  m_cur  = results[i]
        delta_label = f"delta ({l_cur.split()[0]} - {l_prev.split()[0]})"
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
        "  OOD    : Win7AD-1 (hard) + Echo (easy) + Kelihos (floor) — never in training",
        "  (native Zeek conn.log — NO synthetic field mapping)",
    ]

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
        for src in ALL_SOURCES:
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


# ── Main ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    regen       = "--regen"       in sys.argv
    ood_only    = "--ood"         in sys.argv
    no_behavior = "--no-behavior" in sys.argv
    host_pass2  = "--host-pass2"  in sys.argv   # off by default (null result; see thesis_notes_12)

    # ── Sample loading / cache management ───────────────────────────────────────
    if regen and ood_only:
        # Refresh only OOD samples; leave other source caches intact.
        samples = regen_ood_samples()
    elif regen:
        samples = generate_samples()
    elif os.path.exists(CACHE_FILE):
        print(f"[CACHE] Loading {CACHE_FILE}")
        with open(CACHE_FILE) as f:
            samples = json.load(f)
        print(f"  {len(samples)} samples loaded")
    else:
        samples = generate_samples()

    # ── Filter to OOD only if --ood ──────────────────────────────────────────────
    if ood_only:
        samples = [s for s in samples if s["source"] in OOD_SOURCES]
        if not samples:
            print("[ERROR] No OOD samples found in cache. Run with --regen --ood first.")
            sys.exit(1)
        print(f"[OOD] Running inference on {len(samples)} OOD samples only "
              f"(win7ad + sme11 + botnet3).")

    # ── Behavior prompts ─────────────────────────────────────────────────────────
    if no_behavior:
        print("[MODE] --no-behavior enabled: using conn-only prompts.")
    else:
        samples = rebuild_prompts_with_behavior(samples)

    _tokenizer = load_tokenizer(BASE_MODEL)

    ts     = datetime.now().strftime("%Y-%m-%d %H:%M")
    atk_n  = sum(1 for s in samples if s["ground_truth"] == "ATTACK")
    ben_n  = len(samples) - atk_n
    ood_n  = sum(1 for s in samples if s["source"] in OOD_SOURCES)
    mode   = "OOD-ONLY" if ood_only else "FULL"
    out_lines = [
        f"REAL-WORLD ZEEK BENCHMARK [{mode}] — {ts}",
        f"Sources: IoT-23, CTU-13, UWF-ZeekData24, CTU-Normal | OOD: Win7AD-1, Echo, Kelihos",
        f"Samples: {len(samples)} total ({atk_n} attacks / {ben_n} benign) | OOD: {ood_n} | Seed: {RANDOM_SEED}",
    ]
    results     = []
    json_output = {
        "timestamp": datetime.now().isoformat(),
        "mode":      mode,
        "samples":   len(samples),
        "sources":   {src: sum(1 for s in samples if s["source"] == src)
                      for src in ALL_SOURCES},
        "models":    [],
    }

    for label, adapter_path in MODELS:
        if not os.path.isdir(adapter_path):
            print(f"\n[SKIP] {label} — not found at {adapter_path}")
            continue

        model                    = load_model(adapter_path)
        # Auto-detect how this model was trained and serve the matching system
        # prompt (verdict-only for --no-reason adapters), then announce it.
        system_prompt, run_info  = resolve_system_prompt(adapter_path)
        prov                     = provenance_lines(label, adapter_path, run_info, system_prompt)
        for line in prov:
            print(line)
        out_lines.extend(prov)
        preds, unknowns, reasons = run_inference(model, samples, label, system_prompt)
        print_report(preds, samples, label, unknowns, out_lines)
        m                        = compute_metrics(preds, samples, unknowns)
        results.append((label, m))

        # ── Host Pass-2 (host-level aggregation) — OFF by default (--host-pass2) ──
        # Null result: aggregating per-flow predictions into a host verdict does NOT
        # beat per-flow classification (MCC ~0.03, FP recall ~33% — flags most benign
        # hosts). Kept behind a flag for reproducibility; see thesis_notes_12.txt.
        host_samples = []
        host_m = {"accuracy": 0.0, "atk_recall": 0.0, "ben_recall": 0.0, "fmt_fail": 0.0, "mcc": 0.0}
        if host_pass2:
            host_samples = build_host_benchmark_samples(samples, preds)
            if host_samples:
                host_preds, host_unknowns, _ = run_inference(
                    model, host_samples, f"{label} [host pass-2]", system_prompt)
                host_m = compute_metrics(host_preds, host_samples, host_unknowns)
            else:
                host_unknowns = 0
            host_lines = [
                f"\n--- Host Pass-2 ---",
                f"  Hosts           : {len(host_samples)}",
                f"  Format failures : {host_unknowns} ({100*host_unknowns/max(len(host_samples),1):.1f}%)",
                f"  Accuracy        : {host_m['accuracy']:.1%}",
                f"  Atk Recall      : {host_m['atk_recall']:.1%}",
                f"  FP Recall       : {host_m['ben_recall']:.1%}",
                f"  MCC             : {host_m['mcc']:+.4f}",
            ]
            for line in host_lines:
                print(line)
            out_lines.extend(host_lines)

        src_metrics = {}
        for src in ALL_SOURCES:
            idx = [i for i, s in enumerate(samples) if s["source"] == src]
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

        # Per-sample OOD records for gap analysis and failure table.
        # Covers all three OOD probes; includes "source" for per-probe filtering.
        ood_sample_records = [
            {
                "source":       s["source"],
                "raw_label":    s["raw_label"],
                "conn_state":   s.get("conn_state"),
                "proto":        s.get("proto"),
                "resp_p":       s.get("resp_p"),
                "service":      s.get("service"),
                "orig_bytes":   s.get("orig_bytes"),
                "resp_bytes":   s.get("resp_bytes"),
                "ground_truth": s["ground_truth"],
                "prediction":   preds[i],
                "reason":       reasons[i],
            }
            for i, s in enumerate(samples)
            if s["source"] in OOD_SOURCES
        ]

        json_output["models"].append({
            "label":      label,
            "accuracy":   round(m["accuracy"],   4),
            "atk_recall": round(m["atk_recall"], 4),
            "ben_recall": round(m["ben_recall"], 4),
            "fmt_fail":   round(m["fmt_fail"],   4),
            "mcc":        round(m["mcc"],        4),
            "ood_samples": ood_sample_records,
            "host_pass2": {
                "n":          len(host_samples),
                "accuracy":   round(host_m["accuracy"],   4),
                "atk_recall": round(host_m["atk_recall"], 4),
                "ben_recall": round(host_m["ben_recall"], 4),
                "fmt_fail":   round(host_m["fmt_fail"],   4),
                "mcc":        round(host_m["mcc"],        4),
            },
            "per_source": src_metrics,
        })

        # ── Link result back into the adapter's run.json (canonical run only) ──
        # Skip OOD-only / no-behavior passes so they can't overwrite the headline
        # MCC; base-model rows (no run.json) are skipped silently inside.
        if mode == "FULL" and not no_behavior:
            try:
                from ids.run_manifest import attach_benchmark_result
                attach_benchmark_result(adapter_path, {
                    "timestamp":  json_output["timestamp"],
                    "mode":       mode,
                    "samples":    len(samples),
                    "mcc":        round(m["mcc"],        4),
                    "accuracy":   round(m["accuracy"],   4),
                    "atk_recall": round(m["atk_recall"], 4),
                    "ben_recall": round(m["ben_recall"], 4),
                    "fmt_fail":   round(m["fmt_fail"],   4),
                    "per_source": src_metrics,
                })
            except Exception as e:
                print(f"[WARN] could not attach benchmark to run.json: {e}")

        del model
        torch.cuda.empty_cache()

    print_comparison_table(results, json_output, out_lines)

    with open(REPORT_TXT, "w") as f:
        f.write("\n".join(out_lines))
    with open(RESULTS_JSON, "w") as f:
        json.dump(json_output, f, indent=2)

    print(f"\n✅ Report  → {REPORT_TXT}")
    print(f"✅ Results → {RESULTS_JSON}")
