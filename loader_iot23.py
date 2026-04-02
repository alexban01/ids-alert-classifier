"""
loader_iot23.py — Load IoT-23 conn.log.labeled samples from the tar.gz archive.

IoT-23: native Zeek conn.log with 21 tab-separated fields. The last field
bundles "tunnel_parents label detailed-label" as space-separated sub-tokens.
Labels are "Malicious" → ATTACK and "Benign" → FALSE POSITIVE.
"""

import os
import tarfile

from behavior_features import build_behavior_contexts
from preprocess_config import IOT23_BENIGN_CAP, MAX_PER_SOURCE_CLASS
from preprocess_sample import make_sample


def load_iot23(archive_path):
    """Read IoT-23 conn.log.labeled files from tar.gz archive."""
    if not os.path.isfile(archive_path):
        print(f"[SKIP] IoT-23 archive not found: {archive_path}")
        return []

    samples = {"ATTACK": [], "FALSE POSITIVE": []}
    print(f"[IoT-23] Opening archive {archive_path} ...")

    with tarfile.open(archive_path, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.name.endswith("conn.log.labeled") and m.isfile()]
        print(f"  Found {len(members)} conn.log.labeled files")

        for member in members:
            f = tf.extractfile(member)
            if f is None:
                continue
            rows = []
            row_cap = (MAX_PER_SOURCE_CLASS + IOT23_BENIGN_CAP) * 4
            attacks = benign = 0

            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                # Standard Zeek conn.log has 21 tab-separated fields.
                # The last tab field bundles "tunnel_parents label detailed-label"
                # as space-separated sub-tokens (IoT-23 specific formatting).
                if len(parts) < 21:
                    continue
                try:
                    orig_port  = parts[3]
                    resp_port  = parts[5]
                    proto      = parts[6]
                    service    = parts[7]
                    duration   = parts[8]
                    orig_bytes = parts[9]
                    resp_bytes = parts[10]
                    conn_state = parts[11]
                    orig_pkts  = parts[16]
                    resp_pkts  = parts[18]
                    # Last field: "tunnel_parents   label   detailed-label"
                    last = parts[-1]
                    if "Malicious" in last:
                        verdict = "ATTACK"
                    elif "Benign" in last:
                        verdict = "FALSE POSITIVE"
                    else:
                        continue
                except IndexError:
                    continue

                rows.append({
                    "ts":         parts[0],
                    "orig_h":     parts[2],
                    "orig_p":     orig_port,
                    "resp_h":     parts[4],
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
                if verdict == "ATTACK": attacks += 1
                else:                   benign  += 1

                if len(rows) >= row_cap:
                    break

            behavior_ctxs = build_behavior_contexts(rows)
            for row, behavior_ctx in zip(rows, behavior_ctxs):
                cap    = IOT23_BENIGN_CAP if row["verdict"] == "FALSE POSITIVE" else MAX_PER_SOURCE_CLASS
                bucket = samples[row["verdict"]]
                if len(bucket) < cap:
                    bucket.append(make_sample(
                        row["proto"], row["duration"], row["orig_pkts"], row["resp_pkts"],
                        row["orig_bytes"], row["resp_bytes"], row["conn_state"],
                        row["verdict"], "iot23", service=row["service"],
                        resp_port=row["resp_p"], orig_port=row["orig_p"],
                        behavior_ctx=behavior_ctx,
                    ))

            print(f"    {member.name}: {attacks} attacks, {benign} benign")

    print(f"  IoT-23 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]
