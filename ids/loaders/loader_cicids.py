"""
loader_cicids.py — Load CICIDS2017 samples from CICFlowMeter CSVs.

NOTE: CICIDS2017 is DISABLED in v7+ because CICFlowMeter produces
conn_state="-" and proto="unknown" for every flow (both of the most
discriminative Zeek features are always missing).  This loader is
kept for historical benchmarking only — do not re-enable in main().
"""

import glob
import os

import pandas as pd

from ids.preprocess_config import MAX_PER_SOURCE_CLASS
from ids.preprocess_sample import make_sample

# CICFlowMeter → Zeek column mapping
_COLMAP = {
    "Protocol":                    "proto",
    "Flow Duration":               "duration_us",   # microseconds — converted below
    "Total Fwd Packets":           "orig_pkts",
    "Total Backward Packets":      "resp_pkts",
    "Total Length of Fwd Packets": "orig_bytes",
    "Total Length of Bwd Packets": "resp_bytes",
    "Label":                       "label",
}


def load_cicids(base_dir):
    """Read CICIDS2017 CSVs — map CICFlowMeter columns to Zeek schema."""
    csv_files = glob.glob(os.path.join(base_dir, "*.pcap_ISCX.csv"))
    if not csv_files:
        print(f"[SKIP] No CICIDS2017 CSVs found in {base_dir}")
        return []

    # Cap per file so no single CSV (e.g. DDoS) dominates the whole source budget.
    # 8 files × 10k = 80k max per class across CICIDS2017.
    PER_FILE_CAP = 10_000

    print(f"[CICIDS2017] Loading {len(csv_files)} CSV(s) ...")
    samples = {"ATTACK": [], "FALSE POSITIVE": []}

    for fpath in csv_files:
        print(f"  Reading {os.path.basename(fpath)} ...")
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        df.columns = df.columns.str.strip()
        df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()

        avail = {v: k for k, v in _COLMAP.items() if k in df.columns}
        if "label" not in avail:
            continue

        file_counts = {"ATTACK": 0, "FALSE POSITIVE": 0}
        attacks = benign = 0
        for _, row in df.iterrows():
            raw_label = str(row[avail["label"]]).strip()
            verdict   = "FALSE POSITIVE" if raw_label == "BENIGN" else "ATTACK"
            if file_counts[verdict] >= PER_FILE_CAP:
                continue
            bucket = samples[verdict]
            if len(bucket) >= MAX_PER_SOURCE_CLASS:
                continue

            proto  = str(int(float(row[avail["proto"]]))) if "proto" in avail else "unknown"
            dur_us = row[avail["duration_us"]] if "duration_us" in avail else 0
            try:
                duration = str(float(dur_us) / 1e6)  # µs → seconds
            except (ValueError, TypeError):
                duration = "0"

            bucket.append(make_sample(
                proto      = proto,
                duration   = duration,
                orig_pkts  = str(row[avail["orig_pkts"]])  if "orig_pkts"  in avail else "0",
                resp_pkts  = str(row[avail["resp_pkts"]])  if "resp_pkts"  in avail else "0",
                orig_bytes = str(row[avail["orig_bytes"]]) if "orig_bytes" in avail else "0",
                resp_bytes = str(row[avail["resp_bytes"]]) if "resp_bytes" in avail else "0",
                conn_state = "-",   # CICFlowMeter has no conn_state equivalent
                verdict    = verdict,
                source     = "cicids",
            ))
            file_counts[verdict] += 1
            if verdict == "ATTACK": attacks += 1
            else:                   benign  += 1

        print(f"    {attacks} attacks, {benign} benign")

    print(f"  CICIDS2017 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]
