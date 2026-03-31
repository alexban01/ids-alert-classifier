"""
compare_binetflow.py — Cross-reference a Zeek conn.log against a Stratosphere
binetflow file to evaluate model predictions against ground truth labels.

Binetflow format (Argus/Stratosphere):
  StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,...,Label

Two labelling conventions are handled:
  - CTU-13:    From-Botnet-* = ATTACK, From-Normal-* = FP, Background = skip
  - CTU-Malware-Capture: flow=From-Botnet-* = ATTACK, flow=Background* = FP
    (these captures have no From-Normal rows; background traffic is benign)

Matching: 5-tuple (proto, ip_a, port_a, ip_b, port_b) normalised by sorting the
two (ip, port) pairs — direction-agnostic.

URL downloads are cached locally using the capture directory name as a prefix
(e.g. CTU-Malware-Capture-Botnet-78-2_conn.log) to avoid filename collisions.

Usage:
    .venv/bin/python compare_binetflow.py CONN_LOG_OR_URL BINETFLOW_PATH_OR_URL

Example (Zeus botnet, OOD — not in training data):
    .venv/bin/python compare_binetflow.py \\
        https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-78-2/bro/conn.log \\
        https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-78-2/2014-06-06_capture-win8.binetflow.labeled
"""

import sys
import os
import csv
import random
import urllib.request
import torch
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

from classify_conn_log import (
    parse_conn_log, build_prompts, load_hf_model, classify_hf,
    ADAPTER_DIR,
)

# ── Binetflow helpers ──────────────────────────────────────────────────────────

CAPTURE_DIR = "test_captures"


def _url_to_local(url):
    """Derive a unique local path (under test_captures/) from a URL.

    e.g. .../CTU-Malware-Capture-Botnet-78-2/bro/conn.log
         → test_captures/CTU-Malware-Capture-Botnet-78-2_conn.log
    """
    parts = [p for p in url.rstrip("/").split("/") if p and ":" not in p]
    basename = parts[-1].split("?")[0] if parts else "download"
    parent = parts[-2] if len(parts) >= 2 else ""
    # Skip common subdirectory names; use the capture directory instead
    if parent.lower() in ("bro", "data", "logs", "pcap"):
        parent = parts[-3] if len(parts) >= 3 else parent
    filename = f"{parent}_{basename}" if parent else basename
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    return os.path.join(CAPTURE_DIR, filename)


def _norm_key(proto, ip_a, port_a, ip_b, port_b):
    """Normalise 5-tuple so (A→B) and (B→A) produce the same key."""
    pair_a = (ip_a.strip(), str(port_a).strip())
    pair_b = (ip_b.strip(), str(port_b).strip())
    lo, hi = (pair_a, pair_b) if pair_a <= pair_b else (pair_b, pair_a)
    return (proto.strip().lower(), lo[0], lo[1], hi[0], hi[1])


def binetflow_label(raw_label):
    """Map raw Stratosphere label → ATTACK / FALSE POSITIVE / None (skip)."""
    l = raw_label.strip().lower()
    if "malware" in l or "botnet" in l:
        return "ATTACK"
    if "normal" in l or "background" in l:
        return "FALSE POSITIVE"
    return None  # truly unlabelled — skip


def load_binetflow(path_or_url):
    """Download (if URL) and parse binetflow CSV.

    Returns dict: normalised_5tuple → label
    Only rows with a definitive Normal or Malware label are included.
    If the same 5-tuple appears with conflicting labels, ATTACK wins.
    """
    if path_or_url.startswith("http"):
        local = _url_to_local(path_or_url)
        if not os.path.isfile(local):
            print(f"Downloading {path_or_url} → {local} ...")
            urllib.request.urlretrieve(path_or_url, local)
            print(f"  Downloaded ({os.path.getsize(local) // 1024} KB)")
        else:
            print(f"Using cached {local}")
        path = local
    else:
        path = path_or_url

    lookup = {}
    counts = defaultdict(int)

    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if header is None:
                header = [h.strip() for h in row]
                needed = {"Proto", "SrcAddr", "Sport", "DstAddr", "Dport", "Label"}
                if not needed.issubset(set(header)):
                    raise ValueError(f"Missing columns: {needed - set(header)}\nGot: {header}")
                idx = {h: i for i, h in enumerate(header)}
                continue

            if len(row) <= max(idx["Proto"], idx["SrcAddr"], idx["Sport"],
                               idx["DstAddr"], idx["Dport"], idx["Label"]):
                continue

            proto     = row[idx["Proto"]].strip().lower()
            src_ip    = row[idx["SrcAddr"]].strip()
            sport     = row[idx["Sport"]].strip()
            dst_ip    = row[idx["DstAddr"]].strip()
            dport     = row[idx["Dport"]].strip()
            raw_label = row[idx["Label"]].strip()

            label = binetflow_label(raw_label)
            counts[label if label else "Background"] += 1
            if label is None:
                continue

            key = _norm_key(proto, src_ip, sport, dst_ip, dport)
            if key not in lookup or label == "ATTACK":
                lookup[key] = label

    print(f"  Binetflow: {counts['ATTACK']} attacks, "
          f"{counts['FALSE POSITIVE']} normal, "
          f"{counts['Background']} background (skipped)")
    return lookup


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    args = sys.argv[1:]
    max_flows = 1000
    if "--max-flows" in args:
        i = args.index("--max-flows")
        max_flows = int(args[i + 1])
        args = args[:i] + args[i + 2:]

    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    conn_log_path  = args[0]
    binetflow_path = args[1]

    # ── Download conn.log if URL ───────────────────────────────────────────────
    if conn_log_path.startswith("http"):
        local = _url_to_local(conn_log_path)
        if not os.path.isfile(local):
            print(f"Downloading {conn_log_path} → {local} ...")
            urllib.request.urlretrieve(conn_log_path, local)
            print(f"  Downloaded ({os.path.getsize(local) // 1024} KB)")
        else:
            print(f"Using cached {local}")
        conn_log_path = local

    # ── Parse conn.log ─────────────────────────────────────────────────────────
    print(f"\nParsing {conn_log_path} ...")
    rows = parse_conn_log(conn_log_path)
    print(f"  {len(rows)} connections")

    # ── Load binetflow ─────────────────────────────────────────────────────────
    print("\nLoading binetflow ...")
    gt_lookup = load_binetflow(binetflow_path)
    print(f"  {len(gt_lookup)} labelled 5-tuples")

    # ── Match ──────────────────────────────────────────────────────────────────
    matched_rows = []
    matched_gt   = []
    unmatched    = 0

    for r in rows:
        key   = _norm_key(r["proto"], r["orig_h"], r["orig_p"], r["resp_h"], r["resp_p"])
        label = gt_lookup.get(key)
        if label is None:
            unmatched += 1
            continue
        matched_rows.append(r)
        matched_gt.append(label)

    gt_atk = matched_gt.count("ATTACK")
    gt_ben = matched_gt.count("FALSE POSITIVE")
    print(f"\nMatched {len(matched_rows)} / {len(rows)} conn.log rows to binetflow labels")
    print(f"  Ground truth: {gt_atk} attacks, {gt_ben} benign")
    print(f"  Unmatched (no binetflow entry): {unmatched}")

    if not matched_rows:
        print("\nNo matches — check that conn.log and binetflow are from the same capture.")
        sys.exit(1)

    # ── Stratified subsample ───────────────────────────────────────────────────
    if len(matched_rows) > max_flows:
        atk_idx = [i for i, g in enumerate(matched_gt) if g == "ATTACK"]
        ben_idx = [i for i, g in enumerate(matched_gt) if g == "FALSE POSITIVE"]
        random.seed(42)
        n_atk = min(len(atk_idx), max_flows // 2)
        n_ben = min(len(ben_idx), max_flows - n_atk)
        keep  = random.sample(atk_idx, n_atk) + random.sample(ben_idx, n_ben)
        matched_rows = [matched_rows[i] for i in keep]
        matched_gt   = [matched_gt[i]   for i in keep]
        print(f"  Subsampled to {len(matched_rows)} ({n_atk} attacks / {n_ben} benign)"
              f"  [--max-flows {max_flows}]")

    # ── Inference ──────────────────────────────────────────────────────────────
    print(f"\nLoading model ({ADAPTER_DIR}) ...")
    model, tokenizer = load_hf_model()

    prompts = build_prompts(matched_rows)
    print(f"\nRunning inference on {len(prompts)} matched flows ...")
    results     = classify_hf(model, tokenizer, prompts)
    predictions = [v for v, _ in results]

    # ── Metrics ────────────────────────────────────────────────────────────────
    labels_order = ["ATTACK", "FALSE POSITIVE"]
    fmt_fail     = predictions.count("UNKNOWN")
    mcc          = matthews_corrcoef(matched_gt, predictions)

    print(f"\n\n{'='*70}")
    print(f"  MODEL  : {ADAPTER_DIR}")
    print(f"  Capture: {conn_log_path}")
    print(f"  Matched: {len(matched_rows)} flows  ({gt_atk} attacks / {gt_ben} benign)")
    print(f"  Format failures: {fmt_fail} ({fmt_fail/len(predictions)*100:.1f}%)")
    print(f"  MCC    : {mcc:+.4f}")
    print(f"{'='*70}")
    print(classification_report(matched_gt, predictions, labels=labels_order, zero_division=0))

    cm = confusion_matrix(matched_gt, predictions, labels=labels_order)
    print("Confusion Matrix  (rows = actual, cols = predicted)")
    print(f"{'':28s}  {'ATTACK':>8s}  {'FALSE POSITIVE':>14s}")
    for i, row_label in enumerate(labels_order):
        total = cm[i].sum()
        print(f"  {row_label:<26s}", end="")
        for j in range(len(labels_order)):
            pct = f"{cm[i][j]/total*100:.0f}%" if total else "  -"
            print(f"  {cm[i][j]:>5d} ({pct:>4s})", end="")
        print()

    # ── Disagreements ─────────────────────────────────────────────────────────
    fn_list = []  # missed attacks
    fp_list = []  # false alarms

    for r, gt, (pred, raw) in zip(matched_rows, matched_gt, results):
        reason = next((ln[7:].strip() for ln in raw.splitlines()
                       if ln.upper().startswith("REASON:")), "")
        entry = (r, reason)
        if pred == "FALSE POSITIVE" and gt == "ATTACK":
            fn_list.append(entry)
        elif pred == "ATTACK" and gt == "FALSE POSITIVE":
            fp_list.append(entry)

    def _fmt(r, reason):
        return (f"  {r['orig_h']}:{r['orig_p']} → {r['resp_h']}:{r['resp_p']}"
                f"  {r['proto']} {r['conn_state']}"
                f"  dur={r['duration']}  orig={r['orig_bytes']} resp={r['resp_bytes']}"
                + (f"\n    Reason: {reason}" if reason else ""))

    if fn_list:
        print(f"\n── Missed attacks (false negatives): {len(fn_list)} ─────────────────────")
        for r, reason in fn_list[:20]:
            print(_fmt(r, reason))

    if fp_list:
        print(f"\n── False alarms (false positives): {len(fp_list)} ──────────────────────")
        for r, reason in fp_list[:20]:
            print(_fmt(r, reason))
