"""
loader_ctu13.py — Load CTU-13 binetflow samples from the extracted dataset directory.

Expects datasets/ctu-13/CTU-13-Dataset/ to be pre-extracted from the tar.bz2.
Each scenario subdirectory (1–13) contains one .binetflow file.

CTU-13 uses Argus binetflow state notation, which is mapped to Zeek conn_state
equivalents so the model only ever sees Zeek states during training.

Label mapping: Botnet → ATTACK, Normal → FALSE POSITIVE, Background → skip.
"""

import os

from ids.preprocess_config import CTU13_FILE_CAP
from ids.preprocess_sample import make_sample

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


def load_ctu13_file(filepath):
    """Parse a single CTU-13 binetflow file and return samples.

    Called as an independent parallel job from preprocess_zeek.py — each
    scenario file runs in its own process.
    """
    samples = {"ATTACK": [], "FALSE POSITIVE": []}
    attacks = benign = 0

    with open(filepath, errors="replace") as f:
        header = None
        for line in f:
            line = line.strip()
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
            if len(bucket) < CTU13_FILE_CAP:
                bucket.append(make_sample(
                    proto, duration, half, half,
                    src_bytes, dst_bytes, conn_state, verdict, "ctu13",
                    service="-",  # binetflow has no app-layer service field
                    resp_port=resp_port, orig_port=orig_port,
                ))
            if verdict == "ATTACK": attacks += 1
            else:                   benign  += 1

    print(f"    {os.path.basename(filepath)}: {attacks} attacks, {benign} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]


def load_ctu13(dataset_dir):
    """Walk extracted CTU-13-Dataset directory and parse all binetflow files sequentially.

    Sequential convenience wrapper — used for standalone testing only.
    preprocess_zeek.py dispatches each file as an independent parallel job
    via load_ctu13_file(); this wrapper is not called during normal preprocessing.
    """
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] CTU-13 directory not found: {dataset_dir}")
        return []

    binetflow_files = sorted(
        os.path.join(root, fname)
        for root, _, files in os.walk(dataset_dir)
        for fname in files
        if fname.endswith(".binetflow")
    )
    if not binetflow_files:
        print(f"[SKIP] No .binetflow files found in {dataset_dir}")
        return []

    print(f"[CTU-13] Found {len(binetflow_files)} binetflow files in {dataset_dir}")
    all_samples = []
    for fp in binetflow_files:
        all_samples.extend(load_ctu13_file(fp))

    attacks = sum(1 for s in all_samples if s["verdict"] == "ATTACK")
    benign  = sum(1 for s in all_samples if s["verdict"] == "FALSE POSITIVE")
    print(f"  CTU-13 total: {attacks} attacks, {benign} benign")
    return all_samples
