"""
loader_ctu_normal.py — Load CTU-Normal benign-only Zeek conn.log files.

CTU-Normal captures (CTU-Normal-20 through 32) are benign-only Zeek conn.log
TSV files from Stratosphere Lab.  All entries are labelled FALSE POSITIVE.

Standard 21-field TSV format identical to IoT-23.  Uses "-" for unset fields.
"""

import glob
import os

from behavior_features import build_behavior_contexts
from preprocess_config import CTU_NORMAL_CAP
from preprocess_sample import make_sample
from zeek_log_utils import conn_row_from_parts


def load_ctu_normal(dataset_dir):
    """Read CTU-Normal benign Zeek conn.log files (standard 21-field TSV)."""
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] CTU-Normal directory not found: {dataset_dir}")
        return []

    log_files = sorted(glob.glob(os.path.join(dataset_dir, "*.log")))
    if not log_files:
        print(f"[SKIP] No .log files found in {dataset_dir}")
        return []

    print(f"[CTU-Normal] Loading {len(log_files)} conn.log file(s) from {dataset_dir}")
    samples = []
    total   = 0

    for fpath in log_files:
        count   = 0
        rows    = []
        row_cap = max(CTU_NORMAL_CAP - len(samples), 0)
        buffered_benign = 0

        with open(fpath) as f:
            _batch = 0
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 21:
                    continue

                if len(samples) >= CTU_NORMAL_CAP:
                    break

                rows.append(conn_row_from_parts(parts))
                buffered_benign += 1
                count += 1

                if len(rows) >= row_cap:
                    ben_needed = max(0, CTU_NORMAL_CAP - len(samples))
                    if buffered_benign >= ben_needed:
                        break

        behavior_ctxs = build_behavior_contexts(rows)
        for row, behavior_ctx in zip(rows, behavior_ctxs):
            samples.append(make_sample(
                row["proto"], row["duration"], row["orig_pkts"], row["resp_pkts"],
                row["orig_bytes"], row["resp_bytes"], row["conn_state"],
                "FALSE POSITIVE", "ctu_normal", service=row["service"],
                resp_port=row["resp_p"], orig_port=row["orig_p"],
                behavior_ctx=behavior_ctx,
            ))

        total += count
        print(f"  {os.path.basename(fpath)}: {count} benign entries")

    print(f"  CTU-Normal total: {len(samples)} benign")
    return samples
