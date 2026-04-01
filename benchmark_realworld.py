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
import csv
import re
import json
import random
import tarfile
import glob
import urllib.request
import torch
import pandas as pd
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

from behavior_features import build_behavior_contexts, build_host_summaries
from prompt_utils import SYSTEM_PROMPT, build_prompt, build_host_prompt, extract_verdict

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_FILE   = "results/benchmark_realworld_cache.json"
REPORT_TXT   = "results/benchmark_realworld_report.txt"
RESULTS_JSON = "results/benchmark_realworld_results.json"
MAX_NEW_TOKENS = 80
BATCH_SIZE     = 8
CAP            = 300      # max samples per (source, class)
RANDOM_SEED    = 42

MODELS = [
    # ("v4 Fine-tuned",        "./v4-ids-lora-adapter"),
    # ("v6 Fine-tuned",        "./v6-ids-lora-adapter"),
    # ("v7.1 Fine-tuned",      "./v7.1-ids-lora-adapter"),
    # ("v8 ckpt-1500 (ep1)",   "./v8-ids-model/checkpoint-1500"),
    # ("v8.1 Fine-tuned",      "./v8.1-ids-lora-adapter"),
    ("v9.0 Fine-tuned",      "./v9.0-ids-lora-adapter"),
]

DATASETS = {
    "iot23":      "datasets/iot-23/iot_23_datasets_small.tar.gz",
    "ctu13":      "datasets/ctu-13/CTU-13-Dataset.tar.bz2",
    "uwf":        "datasets/uwf-zeekdata24/",
    "ctu_normal": "datasets/ctu-normal/",
}

# OOD regression test — Botnet-3 (Kelihos) is intentionally never in training data.
# v8.1 baseline: MCC +0.06 (near-random). Target for v9.0: MCC > +0.50.
CTU_BOTNET3_BASE_URL = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-3"
CTU_CAPTURE_DIR      = "test_captures"

# ── Sample helpers ──────────────────────────────────────────────────────────────
def make_sample(proto, duration, orig_pkts, resp_pkts,
                orig_bytes, resp_bytes, conn_state,
                ground_truth, source, raw_label, service="-",
                resp_port="-", orig_port="-", ts=None,
                orig_h=None, resp_h=None, uid=None, group_id=None):
    return {
        "prompt":       build_prompt(proto, duration, orig_pkts, resp_pkts,
                                     orig_bytes, resp_bytes, conn_state, service,
                                     resp_port=resp_port, orig_port=orig_port),
        "ground_truth": ground_truth,
        "source":       source,
        "raw_label":    raw_label,
        "ts":           ts,
        "uid":          uid,
        "orig_h":       orig_h,
        "orig_p":       orig_port,
        "resp_h":       resp_h,
        "resp_p":       resp_port,
        "proto":        proto,
        "service":      service,
        "duration":     duration,
        "orig_pkts":    orig_pkts,
        "resp_pkts":    resp_pkts,
        "orig_bytes":   orig_bytes,
        "resp_bytes":   resp_bytes,
        "conn_state":   conn_state,
        "group_id":     group_id or source,
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
                        orig_port  = parts[3],
                        resp_port  = parts[5],
                        ts         = parts[0],
                        uid        = parts[1],
                        orig_h     = parts[2],
                        resp_h     = parts[4],
                        group_id   = member.name,
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
                    orig_port  = row.get("Sport", "-").strip(),
                    resp_port  = row.get("Dport", "-").strip(),
                    orig_h     = row.get("SrcAddr", "").strip(),
                    resp_h     = row.get("DstAddr", "").strip(),
                    group_id   = member.name,
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
                orig_port  = _clean(row.get("id.orig_p", row.get("orig_p", row.get("src_port_zeek", "-")))) or "-",
                resp_port  = _clean(row.get("id.resp_p", row.get("resp_p", row.get("dest_port_zeek", "-")))) or "-",
                ts         = _clean(row.get("ts", row.get("timestamp", ""))) or None,
                uid        = _clean(row.get("uid", "")) or None,
                orig_h     = _clean(row.get("id.orig_h", row.get("orig_h", row.get("src_ip", "")))) or None,
                resp_h     = _clean(row.get("id.resp_h", row.get("resp_h", row.get("dest_ip", "")))) or None,
                group_id   = os.path.basename(fpath),
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
                    orig_port  = parts[3],
                    resp_port  = parts[5],
                    ts         = parts[0],
                    uid        = parts[1],
                    orig_h     = parts[2],
                    resp_h     = parts[4],
                    group_id   = os.path.basename(fpath),
                ))

    print(f"  CTU-Normal: 0 attacks, {len(samples)} benign")
    return samples


# ── CTU-Malware-Botnet-3 (OOD) helpers ─────────────────────────────────────────

def _norm_key(proto, ip_a, port_a, ip_b, port_b):
    """Direction-agnostic 5-tuple key for binetflow ↔ conn.log matching."""
    pair_a = (str(ip_a).strip(), str(port_a).strip())
    pair_b = (str(ip_b).strip(), str(port_b).strip())
    lo, hi = (pair_a, pair_b) if pair_a <= pair_b else (pair_b, pair_a)
    return (str(proto).strip().lower(), lo[0], lo[1], hi[0], hi[1])


def _bench_download(url, local_path):
    """Download url to local_path if not already cached."""
    if os.path.isfile(local_path):
        print(f"  [cache] {os.path.basename(local_path)}")
        return local_path
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    print(f"  Downloading {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(local_path, "wb") as f:
                f.write(resp.read())
        print(f"    {os.path.getsize(local_path) // 1024} KB → {local_path}")
        return local_path
    except Exception as e:
        if os.path.isfile(local_path):
            os.remove(local_path)
        print(f"  [ERROR] Download failed: {e}")
        return None


def _find_binetflow_url(base_url):
    """Fetch directory listing and return the .binetflow.labeled URL."""
    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        matches = re.findall(r'href="([^"/][^"]*\.binetflow(?:\.labeled)?)"', html)
        if matches:
            rel = matches[0]
            return (rel if rel.startswith("http")
                    else base_url.rstrip("/") + "/" + rel.lstrip("/"))
    except Exception as e:
        print(f"  [WARN] Cannot fetch index: {e}")
    return None


def load_ctu_botnet3():
    """Load CTU-Malware-Capture-Botnet-3 (Kelihos) as permanent OOD eval source.

    Downloads conn.log + binetflow from Stratosphere Lab if not cached.
    Label mapping: Botnet → ATTACK, Normal → FALSE POSITIVE, Background → skip.
    This scenario is intentionally NEVER included in training data.
    """
    print(f"\n[CTU-Botnet-3 OOD / Kelihos]")
    os.makedirs(CTU_CAPTURE_DIR, exist_ok=True)

    conn_url      = f"{CTU_BOTNET3_BASE_URL}/bro/conn.log"
    conn_local    = os.path.join(CTU_CAPTURE_DIR, "CTU-Malware-Capture-Botnet-3_conn.log")
    binetflow_url = _find_binetflow_url(CTU_BOTNET3_BASE_URL)

    conn_path = _bench_download(conn_url, conn_local)
    if conn_path is None:
        print("  [SKIP] conn.log download failed")
        return []

    if binetflow_url is None:
        print("  [SKIP] Could not find binetflow URL")
        return []
    binetflow_local = os.path.join(
        CTU_CAPTURE_DIR,
        "CTU-Malware-Capture-Botnet-3_" + binetflow_url.rstrip("/").split("/")[-1],
    )
    binetflow_path = _bench_download(binetflow_url, binetflow_local)
    if binetflow_path is None:
        print("  [SKIP] binetflow download failed")
        return []

    # Build binetflow label lookup
    flow_labels = {}
    try:
        with open(binetflow_path, newline="", errors="replace") as f:
            reader = csv.reader(f)
            header = None
            for row in reader:
                if header is None:
                    header = [h.strip() for h in row]
                    idx    = {h: i for i, h in enumerate(header)}
                    needed = {"Proto", "SrcAddr", "Sport", "DstAddr", "Dport", "Label"}
                    if not needed.issubset(set(header)):
                        print(f"  [SKIP] binetflow missing columns: {needed - set(header)}")
                        return []
                    continue
                if len(row) <= max(idx["Label"], idx["Proto"], idx["SrcAddr"],
                                   idx["Sport"], idx["DstAddr"], idx["Dport"]):
                    continue
                raw_label = row[idx["Label"]].strip().lower()
                if "botnet" in raw_label or "malware" in raw_label:
                    label = "ATTACK"
                elif "normal" in raw_label:
                    label = "FALSE POSITIVE"
                else:
                    continue  # Background → skip
                key = _norm_key(
                    row[idx["Proto"]],
                    row[idx["SrcAddr"]], row[idx["Sport"]],
                    row[idx["DstAddr"]], row[idx["Dport"]],
                )
                if key not in flow_labels or label == "ATTACK":
                    flow_labels[key] = label
    except Exception as e:
        print(f"  [SKIP] binetflow parse error: {e}")
        return []

    atk_n = sum(1 for v in flow_labels.values() if v == "ATTACK")
    ben_n = sum(1 for v in flow_labels.values() if v == "FALSE POSITIVE")
    print(f"  binetflow: {atk_n} ATTACK + {ben_n} FALSE POSITIVE labels")

    # Match conn.log flows to binetflow labels
    buckets = defaultdict(list)
    try:
        with open(conn_path, errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 19:
                    continue
                proto  = parts[6]
                orig_h = parts[2]; orig_p = parts[3]
                resp_h = parts[4]; resp_p = parts[5]
                key    = _norm_key(proto, orig_h, orig_p, resp_h, resp_p)
                label  = flow_labels.get(key)
                if label is None:
                    continue
                if len(buckets[label]) >= CAP:
                    continue
                buckets[label].append(make_sample(
                    proto      = proto,
                    duration   = parts[8],
                    orig_pkts  = parts[16] if len(parts) > 16 else "-",
                    resp_pkts  = parts[18] if len(parts) > 18 else "-",
                    orig_bytes = parts[9],
                    resp_bytes = parts[10],
                    conn_state = parts[11],
                    ground_truth = label,
                    source     = "ctu_botnet3",
                    raw_label  = "Kelihos" if label == "ATTACK" else "Benign",
                    service    = parts[7],
                    orig_port  = orig_p,
                    resp_port  = resp_p,
                    ts         = parts[0],
                    uid        = parts[1],
                    orig_h     = orig_h,
                    resp_h     = resp_h,
                    group_id   = "ctu_botnet3",
                ))
    except Exception as e:
        print(f"  [SKIP] conn.log parse error: {e}")
        return []

    atk = len(buckets["ATTACK"])
    ben = len(buckets["FALSE POSITIVE"])
    print(f"  Botnet-3 (OOD): {atk} attacks, {ben} benign sampled")
    return buckets["ATTACK"] + buckets["FALSE POSITIVE"]


# ── Sample generation ───────────────────────────────────────────────────────────
def generate_samples():
    all_samples = []
    all_samples += load_iot23(DATASETS["iot23"])
    all_samples += load_ctu13(DATASETS["ctu13"])
    all_samples += load_uwf(DATASETS["uwf"])
    all_samples += load_ctu_normal(DATASETS["ctu_normal"])
    # v9.0: CTU-Malware-Botnet-3 (Kelihos) — permanent OOD regression test.
    # Never in training data. v8.1 MCC baseline: +0.06. Target: > +0.50.
    all_samples += load_ctu_botnet3()

    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)

    with open(CACHE_FILE, "w") as f:
        json.dump(all_samples, f, indent=2)

    atk = sum(1 for s in all_samples if s["ground_truth"] == "ATTACK")
    ben = sum(1 for s in all_samples if s["ground_truth"] == "FALSE POSITIVE")
    print(f"\n✅ {len(all_samples)} samples cached → {CACHE_FILE}")
    print(f"   Attacks: {atk}  |  Benign: {ben}")
    return all_samples


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
    "iot23":       "IoT-23         (Zeek conn.log)",
    "ctu13":       "CTU-13         (binetflow)",
    "uwf":         "UWF-ZeekData24 (Zeek conn.log)",
    "ctu_normal":  "CTU-Normal     (Zeek conn.log)",
    "ctu_botnet3": "Botnet-3 [OOD] (Kelihos)",
}

# Sources used in JSON per-source output (training sources + OOD)
ALL_SOURCES = ["iot23", "ctu13", "uwf", "ctu_normal", "ctu_botnet3"]


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

    samples = rebuild_prompts_with_behavior(samples)

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    ts        = datetime.now().strftime("%Y-%m-%d %H:%M")
    atk_n     = sum(1 for s in samples if s["ground_truth"] == "ATTACK")
    ben_n     = len(samples) - atk_n
    ood_n     = sum(1 for s in samples if s["source"] == "ctu_botnet3")
    out_lines = [
        f"REAL-WORLD ZEEK BENCHMARK — {ts}",
        f"Sources: IoT-23, CTU-13, UWF-ZeekData24, CTU-Normal | OOD: Botnet-3 (Kelihos)",
        f"Samples: {len(samples)} total ({atk_n} attacks / {ben_n} benign) | OOD: {ood_n} | Seed: {RANDOM_SEED}",
    ]
    results     = []
    json_output = {
        "timestamp": datetime.now().isoformat(),
        "samples":   len(samples),
        "sources":   {src: sum(1 for s in samples if s["source"] == src)
                      for src in ALL_SOURCES},
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

        host_samples = build_host_benchmark_samples(samples, preds)
        if host_samples:
            host_preds, host_unknowns = run_inference(model, host_samples, f"{label} [host pass-2]")
            host_m = compute_metrics(host_preds, host_samples, host_unknowns)
        else:
            host_unknowns = 0
            host_m = {"accuracy": 0.0, "atk_recall": 0.0, "ben_recall": 0.0, "fmt_fail": 0.0, "mcc": 0.0}
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

        # Per-source metrics for JSON
        src_metrics = {}
        for src in ALL_SOURCES:
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
            "  OOD    : Botnet-3 (Kelihos) — never in training data",
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

    with open(REPORT_TXT, "w") as f:
        f.write("\n".join(out_lines))
    with open(RESULTS_JSON, "w") as f:
        json.dump(json_output, f, indent=2)

    print(f"\n✅ Report  → {REPORT_TXT}")
    print(f"✅ Results → {RESULTS_JSON}")
