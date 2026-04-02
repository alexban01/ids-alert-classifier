"""
loader_ctu13.py — Load CTU-13 binetflow samples from the tar.bz2 archive.

CTU-13 uses Argus binetflow state notation, which is mapped to Zeek conn_state
equivalents so the model only ever sees Zeek states during training.

Label mapping: Botnet → ATTACK, Normal → FALSE POSITIVE, Background → skip.
"""

import os
import tarfile

from preprocess_config import MAX_PER_SOURCE_CLASS
from preprocess_sample import make_sample

# Argus binetflow state → Zeek conn_state
_CTU_STATE_MAP = {
    "INT":       "S1",    # mid-flow established, no FIN seen
    "CON":       "SF",    # completed connection
    "FIN":       "SF",    # completed with FIN
    "FSPA_FSPA": "SF",    # FIN bidirectional = completed
    "FSA_FSA":   "SF",
    "SPA_FSPA":  "SF",
    "PA_PA":     "OTH",   # PSH-ACK only, no SYN seen
    "EST":       "S1",    # established
    "S_":        "S0",    # SYN only, no response
    "REQ":       "S0",
    "SRPA_SPA":  "RSTO",  # RST from originator
    "SRST":      "RSTO",
}


def load_ctu13(archive_path):
    """Read CTU-13 binetflow files from tar.bz2 archive."""
    if not os.path.isfile(archive_path):
        print(f"[SKIP] CTU-13 archive not found: {archive_path}")
        return []

    samples = {"ATTACK": [], "FALSE POSITIVE": []}
    print(f"[CTU-13] Opening archive {archive_path} ...")

    with tarfile.open(archive_path, "r:bz2") as tf:
        members = [m for m in tf.getmembers()
                   if m.name.endswith(".binetflow") and m.isfile()]
        print(f"  Found {len(members)} binetflow files")

        for member in members:
            f = tf.extractfile(member)
            if f is None:
                continue
            attacks = benign = 0
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
                    continue   # skip Background (unlabeled)

                proto      = row.get("Proto", "unknown").strip().lower()
                duration   = row.get("Dur",   "0").strip()
                raw_state  = row.get("State", "-").strip().upper()
                conn_state = _CTU_STATE_MAP.get(raw_state, "-")
                tot_pkts   = row.get("TotPkts",  "0").strip()
                src_bytes  = row.get("SrcBytes", "0").strip()
                tot_bytes  = row.get("TotBytes", "0").strip()
                orig_port  = row.get("Sport", "-").strip()
                resp_port  = row.get("Dport", "-").strip()
                try:
                    dst_bytes = str(float(tot_bytes) - float(src_bytes))
                except ValueError:
                    dst_bytes = "0"
                # binetflow has TotPkts only; split evenly as approximation
                try:
                    half = str(int(float(tot_pkts)) // 2)
                except ValueError:
                    half = "0"

                bucket = samples[verdict]
                if len(bucket) < MAX_PER_SOURCE_CLASS:
                    bucket.append(make_sample(
                        proto, duration, half, half,
                        src_bytes, dst_bytes, conn_state, verdict, "ctu13",
                        service="-",  # binetflow has no app-layer service field
                        resp_port=resp_port, orig_port=orig_port,
                    ))
                if verdict == "ATTACK": attacks += 1
                else:                   benign  += 1

            print(f"    {member.name}: {attacks} attacks, {benign} benign")

    print(f"  CTU-13 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]
