"""
loader_unsw.py — Load UNSW-NB15 samples from parquet or CSV files.

UNSW-NB15 was generated with Bro/Zeek so column names have good overlap, but
differ slightly (e.g. "protocol" vs "proto", "binary_label" vs "label").
Column discovery uses fallback lists to handle both naming conventions.
"""

import glob
import os

import pandas as pd

from behavior_features import build_behavior_contexts
from preprocess_config import MAX_PER_SOURCE_CLASS
from preprocess_sample import make_sample


def load_unsw(dataset_dir):
    """Read UNSW-NB15 from CSV or parquet files."""
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] UNSW-NB15 directory not found: {dataset_dir}")
        return []

    parquet_files = glob.glob(os.path.join(dataset_dir, "**/*.parquet"), recursive=True)
    csv_files     = glob.glob(os.path.join(dataset_dir, "**/*.csv"),     recursive=True)
    # exclude HuggingFace cache metadata and index files
    csv_files = [f for f in csv_files
                 if ".cache" not in f and "metadata" not in f.lower()
                 and not os.path.basename(f).startswith(".")]

    files = parquet_files if parquet_files else csv_files
    if not files:
        print(f"[SKIP] No parquet or CSV files found in {dataset_dir}")
        return []

    print(f"[UNSW-NB15] Loading {len(files)} file(s) from {dataset_dir}")
    samples  = {"ATTACK": [], "FALSE POSITIVE": []}
    row_cap  = (MAX_PER_SOURCE_CLASS + MAX_PER_SOURCE_CLASS) * 4

    for fpath in files:
        print(f"  Reading {os.path.basename(fpath)} ...")
        try:
            df = pd.read_parquet(fpath) if fpath.endswith(".parquet") else pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        df.columns = [c.strip().lower() for c in df.columns]

        proto_col   = next((c for c in ["proto", "protocol"]                         if c in df.columns), None)
        dur_col     = next((c for c in ["dur", "duration"]                           if c in df.columns), None)
        spkts_col   = next((c for c in ["spkts", "orig_pkts"]                        if c in df.columns), None)
        dpkts_col   = next((c for c in ["dpkts", "resp_pkts"]                        if c in df.columns), None)
        sbytes_col  = next((c for c in ["sbytes", "orig_bytes"]                      if c in df.columns), None)
        dbytes_col  = next((c for c in ["dbytes", "resp_bytes"]                      if c in df.columns), None)
        state_col   = next((c for c in ["state", "conn_state"]                       if c in df.columns), None)
        svc_col     = next((c for c in ["service"]                                   if c in df.columns), None)
        sport_col   = next((c for c in ["sport", "source_port", "orig_p"]            if c in df.columns), None)
        dport_col   = next((c for c in ["dport", "destination_port", "resp_p"]       if c in df.columns), None)
        ts_col      = next((c for c in ["ts", "timestamp", "stime", "starttime"]     if c in df.columns), None)
        src_h_col   = next((c for c in ["srcip", "src_ip", "orig_h", "id.orig_h"]    if c in df.columns), None)
        dst_h_col   = next((c for c in ["dstip", "dst_ip", "resp_h", "id.resp_h"]    if c in df.columns), None)
        label_col   = next((c for c in ["binary_label", "label"]                     if c in df.columns), None)

        if label_col is None:
            print(f"    No label column found, skipping.")
            continue

        df = df.replace([float("inf"), float("-inf")], float("nan")).dropna(subset=[label_col])

        rows = []
        attacks = benign = 0

        for _, row in df.iterrows():
            lv = row[label_col]
            try:
                verdict = "ATTACK" if int(float(lv)) == 1 else "FALSE POSITIVE"
            except (ValueError, TypeError):
                verdict = "ATTACK" if str(lv).strip() not in ("0", "Normal", "BENIGN") else "FALSE POSITIVE"

            rows.append({
                "ts":         str(row[ts_col]).strip()     if ts_col     else None,
                "orig_h":     str(row[src_h_col]).strip()  if src_h_col  else None,
                "orig_p":     str(row[sport_col]).strip()  if sport_col  else "-",
                "resp_h":     str(row[dst_h_col]).strip()  if dst_h_col  else None,
                "resp_p":     str(row[dport_col]).strip()  if dport_col  else "-",
                "proto":      str(row[proto_col]).strip()  if proto_col  else "unknown",
                "service":    str(row[svc_col]).strip()    if svc_col    else "-",
                "duration":   str(row[dur_col]).strip()    if dur_col    else "0",
                "orig_bytes": str(row[sbytes_col]).strip() if sbytes_col else "0",
                "resp_bytes": str(row[dbytes_col]).strip() if dbytes_col else "0",
                "conn_state": str(row[state_col]).strip()  if state_col  else "-",
                "orig_pkts":  str(row[spkts_col]).strip()  if spkts_col  else "0",
                "resp_pkts":  str(row[dpkts_col]).strip()  if dpkts_col  else "0",
                "verdict":    verdict,
            })

            if len(rows) >= row_cap:
                break

        behavior_ctxs = build_behavior_contexts(rows)
        for row, behavior_ctx in zip(rows, behavior_ctxs):
            bucket = samples[row["verdict"]]
            if len(bucket) >= MAX_PER_SOURCE_CLASS:
                continue
            bucket.append(make_sample(
                proto      = row["proto"],
                duration   = row["duration"],
                orig_pkts  = row["orig_pkts"],
                resp_pkts  = row["resp_pkts"],
                orig_bytes = row["orig_bytes"],
                resp_bytes = row["resp_bytes"],
                conn_state = row["conn_state"],
                verdict    = row["verdict"],
                source     = "unsw",
                service    = row["service"],
                orig_port  = row["orig_p"],
                resp_port  = row["resp_p"],
                behavior_ctx=behavior_ctx,
            ))
            if row["verdict"] == "ATTACK": attacks += 1
            else:                          benign  += 1

        print(f"    {attacks} attacks, {benign} benign")

    print(f"  UNSW-NB15 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]
