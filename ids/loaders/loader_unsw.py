"""
loader_unsw.py — Load UNSW-NB15 samples from parquet or CSV files.

UNSW-NB15 was generated with Bro/Zeek so column names have good overlap, but
differ slightly (e.g. "protocol" vs "proto", "binary_label" vs "label").
Column discovery uses fallback lists to handle both naming conventions.
"""

import glob
import os

import pandas as pd

from ids.behavior_features import build_behavior_contexts
from ids.preprocess_config import MAX_PER_SOURCE_CLASS
from ids.preprocess_sample import make_sample


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

        # Truncate before any column extraction to avoid processing millions of rows.
        # 2× cap (was 8×) with early-exit once both buckets are full.
        row_cap  = (MAX_PER_SOURCE_CLASS + MAX_PER_SOURCE_CLASS) * 2
        df = df.head(row_cap)

        def _col(name, default):
            if name and name in df.columns:
                return df[name].fillna(default).astype(str).str.strip().tolist()
            return [str(default)] * len(df)

        ts_vals    = _col(ts_col,    "")
        srch_vals  = _col(src_h_col, "")
        dsth_vals  = _col(dst_h_col, "")
        sport_vals = _col(sport_col, "-")
        dport_vals = _col(dport_col, "-")
        proto_vals = _col(proto_col, "unknown")
        svc_vals   = _col(svc_col,   "-")
        dur_vals   = _col(dur_col,   "0")
        sb_vals    = _col(sbytes_col,"0")
        db_vals    = _col(dbytes_col,"0")
        state_vals = _col(state_col, "-")
        sp_vals    = _col(spkts_col, "0")
        dp_vals    = _col(dpkts_col, "0")
        label_vals = df[label_col].tolist()

        rows = []
        attacks = benign = 0

        for i, lv in enumerate(label_vals):
            try:
                verdict = "ATTACK" if int(float(lv)) == 1 else "FALSE POSITIVE"
            except (ValueError, TypeError):
                verdict = "ATTACK" if str(lv).strip() not in ("0", "Normal", "BENIGN") else "FALSE POSITIVE"

            rows.append({
                "ts":         ts_vals[i],
                "orig_h":     srch_vals[i],
                "orig_p":     sport_vals[i],
                "resp_h":     dsth_vals[i],
                "resp_p":     dport_vals[i],
                "proto":      proto_vals[i],
                "service":    svc_vals[i],
                "duration":   dur_vals[i],
                "orig_bytes": sb_vals[i],
                "resp_bytes": db_vals[i],
                "conn_state": state_vals[i],
                "orig_pkts":  sp_vals[i],
                "resp_pkts":  dp_vals[i],
                "verdict":    verdict,
            })
            if verdict == "ATTACK": attacks += 1
            else:                   benign  += 1

            if attacks >= MAX_PER_SOURCE_CLASS and benign >= MAX_PER_SOURCE_CLASS:
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
