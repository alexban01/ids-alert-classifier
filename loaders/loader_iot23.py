"""
loader_iot23.py — Load IoT-23 conn.log.labeled samples from extracted files.

IoT-23: native Zeek conn.log with 21 tab-separated fields. The last field
bundles "tunnel_parents label detailed-label" as space-separated sub-tokens.
Labels are "Malicious" → ATTACK and "Benign" → FALSE POSITIVE.

The archive should be pre-extracted; each conn.log.labeled file is dispatched
as an independent parallel job from preprocess_zeek.py (same pattern as
ctu13_single).  This avoids re-decompressing the 8.7 GB tar.gz on every run.
"""

import os

from behavior_features import build_behavior_contexts
from preprocess_config import IOT23_FILE_ATTACK_CAP, IOT23_FILE_BENIGN_CAP
from preprocess_sample import make_sample
from zeek_log_utils import conn_row_from_parts


def load_iot23_file(filepath):
    """Parse a single IoT-23 conn.log.labeled file and return samples.

    Called as an independent parallel job from preprocess_zeek.py.
    Attack cap: IOT23_FILE_ATTACK_CAP (matches CTU13_FILE_CAP to prevent S0-flood dominance).
    Benign cap: IOT23_FILE_BENIGN_CAP (keeps S0-UDP benign low per file).
    """
    samples = {"ATTACK": [], "FALSE POSITIVE": []}
    rows = []
    row_cap = (IOT23_FILE_ATTACK_CAP + IOT23_FILE_BENIGN_CAP) * 2
    attacks = benign = 0

    with open(filepath, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 21:
                continue
            try:
                last = parts[-1]
                if "Malicious" in last:
                    verdict = "ATTACK"
                elif "Benign" in last:
                    verdict = "FALSE POSITIVE"
                else:
                    continue
            except IndexError:
                continue

            row = conn_row_from_parts(parts)
            row["verdict"] = verdict
            rows.append(row)
            if verdict == "ATTACK": attacks += 1
            else:                   benign  += 1

            if attacks >= IOT23_FILE_ATTACK_CAP and benign >= IOT23_FILE_BENIGN_CAP:
                break
            if len(rows) >= row_cap:
                break

    behavior_ctxs = build_behavior_contexts(rows)
    for row, behavior_ctx in zip(rows, behavior_ctxs):
        cap    = IOT23_FILE_BENIGN_CAP if row["verdict"] == "FALSE POSITIVE" else IOT23_FILE_ATTACK_CAP
        bucket = samples[row["verdict"]]
        if len(bucket) < cap:
            bucket.append(make_sample(
                row["proto"], row["duration"], row["orig_pkts"], row["resp_pkts"],
                row["orig_bytes"], row["resp_bytes"], row["conn_state"],
                row["verdict"], "iot23", service=row["service"],
                resp_port=row["resp_p"], orig_port=row["orig_p"],
                behavior_ctx=behavior_ctx,
            ))

    atk = len(samples["ATTACK"])
    ben = len(samples["FALSE POSITIVE"])
    print(f"    {os.path.basename(os.path.dirname(os.path.dirname(filepath)))}: "
          f"{atk} attacks, {ben} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]


def load_iot23(dataset_dir):
    """Walk extracted IoT-23 directory and return all conn.log.labeled file paths.

    Sequential convenience wrapper for standalone testing only.
    preprocess_zeek.py dispatches each file as an independent parallel job
    via load_iot23_file(); this wrapper is not called during normal preprocessing.
    """
    files = sorted(
        os.path.join(root, fname)
        for root, _, filenames in os.walk(dataset_dir)
        for fname in filenames
        if fname == "conn.log.labeled"
    )
    if not files:
        print(f"[SKIP] No conn.log.labeled files found under {dataset_dir}")
        return []

    print(f"[IoT-23] Found {len(files)} conn.log.labeled files in {dataset_dir}")
    all_samples = []
    for fp in files:
        all_samples.extend(load_iot23_file(fp))

    atk = sum(1 for s in all_samples if s["verdict"] == "ATTACK")
    ben = sum(1 for s in all_samples if s["verdict"] == "FALSE POSITIVE")
    print(f"  IoT-23 total: {atk} attacks, {ben} benign")
    return all_samples
