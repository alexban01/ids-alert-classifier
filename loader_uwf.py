"""
loader_uwf.py — Load UWF-ZeekData24 samples from Spark-output CSV files.

UWF-ZeekData24 is real Zeek conn.log from University of West Florida cyber range.
Columns match Zeek naming.  Empty strings for missing values in S0 connections
are passed through as-is; build_prompt/_safe converts them to N/A.

v8.1: Credential Access (port 4848/ssl) and Defense Evasion (port 445/smb)
attack tactics are included.  Initial Access (port 80 SF — ambiguous web
traffic) and Exfiltration (23 rows, too small) are skipped.
"""

import glob
import os

import pandas as pd

from behavior_features import build_behavior_contexts
from preprocess_config import MAX_PER_SOURCE_CLASS
from preprocess_sample import make_sample

_UWF_ALLOWED_TACTICS = {"Credential Access", "Defense Evasion"}


def load_uwf(dataset_dir):
    """Read UWF-ZeekData24 CSV files (Spark output, one dir per MITRE tactic)."""
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] UWF-ZeekData24 directory not found: {dataset_dir}")
        return []

    csv_files = glob.glob(os.path.join(dataset_dir, "**/*.csv"), recursive=True)
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith(".")]
    if not csv_files:
        print(f"[SKIP] No CSV files found in {dataset_dir}")
        return []

    print(f"[UWF-ZeekData24] Loading {len(csv_files)} CSV(s) from {dataset_dir}")
    samples = {"ATTACK": [], "FALSE POSITIVE": []}
    row_cap = (MAX_PER_SOURCE_CLASS + MAX_PER_SOURCE_CLASS) * 4

    for fpath in csv_files:
        print(f"  Reading {os.path.relpath(fpath, dataset_dir)} ...")
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        df.columns = [c.strip() for c in df.columns]

        label_col = next((c for c in ["label_binary", "label_tactic"] if c in df.columns), None)
        if label_col is None:
            print(f"    No label column found, skipping.")
            continue

        rows = []
        buffered_counts = {"ATTACK": 0, "FALSE POSITIVE": 0}
        attacks = benign = 0

        for _, row in df.iterrows():
            if label_col == "label_binary":
                verdict = "ATTACK" if str(row[label_col]).strip() == "True" else "FALSE POSITIVE"
            else:
                verdict = "ATTACK" if str(row[label_col]).strip() != "none" else "FALSE POSITIVE"

            if verdict == "ATTACK":
                tactic = str(row.get("label_tactic", "")).strip()
                if tactic not in _UWF_ALLOWED_TACTICS:
                    continue

            proto      = str(row.get("proto",      "unknown")).strip()
            service    = str(row.get("service",    "-")).strip()
            duration   = str(row.get("duration",   "")).strip()
            orig_pkts  = str(row.get("orig_pkts",  "")).strip()
            resp_pkts  = str(row.get("resp_pkts",  "")).strip()
            orig_bytes = str(row.get("orig_bytes", "")).strip()
            resp_bytes = str(row.get("resp_bytes", "")).strip()
            conn_state = str(row.get("conn_state", "-")).strip()
            orig_port  = str(row.get("id.orig_p",  row.get("orig_p",  row.get("src_port_zeek",  "-")))).strip()
            resp_port  = str(row.get("id.resp_p",  row.get("resp_p",  row.get("dest_port_zeek", "-")))).strip()

            # pandas converts empty CSV cells to nan
            if service    in ("nan", "None"): service    = "-"
            if duration   in ("nan", "None"): duration   = ""
            if orig_pkts  in ("nan", "None"): orig_pkts  = ""
            if resp_pkts  in ("nan", "None"): resp_pkts  = ""
            if orig_bytes in ("nan", "None"): orig_bytes = ""
            if resp_bytes in ("nan", "None"): resp_bytes = ""
            if orig_port  in ("nan", "None"): orig_port  = "-"
            if resp_port  in ("nan", "None"): resp_port  = "-"

            ts = str(row.get("ts", row.get("timestamp", ""))).strip()
            if ts in ("nan", "None", ""):
                ts = None
            orig_h = str(row.get("id.orig_h", row.get("orig_h", row.get("src_ip", "")))).strip()
            if orig_h in ("nan", "None", ""):
                orig_h = None
            resp_h = str(row.get("id.resp_h", row.get("resp_h", row.get("dest_ip", "")))).strip()
            if resp_h in ("nan", "None", ""):
                resp_h = None

            buffered_counts[verdict] += 1
            rows.append({
                "ts":         ts,
                "orig_h":     orig_h,
                "orig_p":     orig_port,
                "resp_h":     resp_h,
                "resp_p":     resp_port,
                "proto":      proto,
                "service":    service,
                "duration":   duration,
                "orig_bytes": orig_bytes,
                "resp_bytes": resp_bytes,
                "conn_state": conn_state,
                "orig_pkts":  orig_pkts,
                "resp_pkts":  resp_pkts,
                "verdict":    verdict,
            })

            if len(rows) >= row_cap:
                atk_needed = max(0, MAX_PER_SOURCE_CLASS - len(samples["ATTACK"]))
                ben_needed = max(0, MAX_PER_SOURCE_CLASS - len(samples["FALSE POSITIVE"]))
                if (buffered_counts["ATTACK"] >= atk_needed and
                        buffered_counts["FALSE POSITIVE"] >= ben_needed):
                    break

        behavior_ctxs = build_behavior_contexts(rows)
        for row, behavior_ctx in zip(rows, behavior_ctxs):
            bucket = samples[row["verdict"]]
            if len(bucket) >= MAX_PER_SOURCE_CLASS:
                continue
            bucket.append(make_sample(
                row["proto"], row["duration"], row["orig_pkts"], row["resp_pkts"],
                row["orig_bytes"], row["resp_bytes"], row["conn_state"],
                row["verdict"], "uwf", service=row["service"],
                resp_port=row["resp_p"], orig_port=row["orig_p"],
                behavior_ctx=behavior_ctx,
            ))
            if row["verdict"] == "ATTACK": attacks += 1
            else:                          benign  += 1

        print(f"    {attacks} attacks, {benign} benign")

    print(f"  UWF-ZeekData24 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]
