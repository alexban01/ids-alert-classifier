"""
preprocess_zeek.py — Build training JSONL from Zeek-native / Zeek-compatible datasets.

Datasets supported:
  - IoT-23        : native Zeek conn.log.labeled files
  - CTU-13        : binetflow (Argus bidirectional flows)
  - UNSW-NB15     : CSV/parquet (Bro/Zeek-generated, good column overlap)
  - CICIDS2017    : CICFlowMeter CSVs (already present) — mapped to Zeek schema

Output: zeek_dataset.jsonl  (same chat format as ids_dataset.jsonl)

Zeek conn.log feature set used across all sources:
  proto, duration, orig_pkts, resp_pkts, orig_bytes, resp_bytes,
  conn_state, bytes_per_sec, orig_bytes_per_pkt, resp_bytes_per_pkt
"""

import os
import json
import random
import tarfile
import glob
import math

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FILE   = "zeek_dataset.jsonl"
RANDOM_SEED   = 42
MAX_PER_SOURCE_CLASS = 80_000   # cap per (source, class) before merging
FINAL_BENIGN  = 120_000         # target benign rows in final dataset
FINAL_ATTACK  = 180_000         # target attack rows (60/40 split, attacks heavier)

DATASETS = {
    "iot23":    "datasets/iot-23/iot_23_datasets_small.tar.gz",
    "ctu13":    "datasets/ctu-13/CTU-13-Dataset.tar.bz2",
    "unsw":     "datasets/unsw-nb15/",
    "cicids":   ".",   # looks for *.pcap_ISCX.csv in cwd
}

SYSTEM_PROMPT = (
    "You are a network security analyst. "
    "Always respond with VERDICT: <ATTACK or FALSE POSITIVE> on the first line, "
    "followed by REASON: <brief explanation>."
)

# ── Reason generators ──────────────────────────────────────────────────────────
def _safe(v, fmt=".1f"):
    try:
        return format(float(v), fmt) if v not in (None, "", "-", "?") else "N/A"
    except (ValueError, TypeError):
        return "N/A"

ATTACK_REASONS = [
    "Traffic pattern matches known malicious behavior with anomalous packet ratios.",
    "Connection characteristics are inconsistent with normal user activity.",
    "Flow statistics deviate significantly from baseline benign traffic.",
    "Short-duration high-volume flow typical of automated attack tooling.",
    "Asymmetric byte distribution and connection state indicate hostile intent.",
    "Packet timing and volume match scanning or exploitation patterns.",
    "Connection state and byte counts are inconsistent with legitimate use.",
    "Flow metrics match botnet C2 communication profiles.",
    "Unidirectional or near-unidirectional flow with anomalous size is suspicious.",
    "Connection terminated abnormally with byte ratios typical of attack traffic.",
]

BENIGN_REASONS = [
    "Symmetric flow with normal byte and packet ratios consistent with legitimate traffic.",
    "Connection established and closed cleanly; no anomalous volume or timing.",
    "Packet sizes and counts match expected web or file-transfer activity.",
    "Flow duration and byte counts are within normal operational range.",
    "Bidirectional exchange with balanced originator/responder bytes; likely benign.",
    "Connection state indicates normal handshake and teardown sequence.",
    "Traffic volume and protocol use are consistent with standard enterprise behavior.",
    "Flow characteristics match DNS, HTTP, or routine background communication.",
    "Short-lived connection with small byte counts; consistent with keep-alives or health checks.",
    "No anomalous ratios detected; flow matches baseline for this protocol.",
]

random.seed(RANDOM_SEED)

def pick_reason(verdict):
    pool = ATTACK_REASONS if verdict == "ATTACK" else BENIGN_REASONS
    return random.choice(pool)

# ── Feature → prompt ───────────────────────────────────────────────────────────
def build_prompt(proto, duration, orig_pkts, resp_pkts,
                 orig_bytes, resp_bytes, conn_state):
    """Convert Zeek-native features to model prompt text."""
    try:
        dur_f   = float(duration)
        ob_f    = float(orig_bytes)
        rb_f    = float(resp_bytes)
        op_f    = float(orig_pkts)
        rp_f    = float(resp_pkts)
        bps     = (ob_f + rb_f) / dur_f if dur_f > 0 else 0.0
        op_sz   = ob_f / op_f if op_f > 0 else 0.0
        rp_sz   = rb_f / rp_f if rp_f > 0 else 0.0
    except (ValueError, TypeError, ZeroDivisionError):
        bps = op_sz = rp_sz = 0.0

    lines = [
        "Analyze this network connection and classify it as ATTACK or FALSE POSITIVE.\n",
        f"  Proto:              {proto}",
        f"  Duration (s):       {_safe(duration, '.6f')}",
        f"  Orig Packets:       {_safe(orig_pkts, '.0f')}",
        f"  Resp Packets:       {_safe(resp_pkts, '.0f')}",
        f"  Orig Bytes:         {_safe(orig_bytes, '.0f')}",
        f"  Resp Bytes:         {_safe(resp_bytes, '.0f')}",
        f"  Conn State:         {conn_state}",
        f"  Bytes/sec:          {_safe(bps, '.1f')}",
        f"  Orig Bytes/Pkt:     {_safe(op_sz, '.1f')}",
        f"  Resp Bytes/Pkt:     {_safe(rp_sz, '.1f')}",
    ]
    return "\n".join(lines)

def make_sample(proto, duration, orig_pkts, resp_pkts,
                orig_bytes, resp_bytes, conn_state, verdict, source):
    prompt  = build_prompt(proto, duration, orig_pkts, resp_pkts,
                           orig_bytes, resp_bytes, conn_state)
    reason  = pick_reason(verdict)
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": f"VERDICT: {verdict}\nREASON: {reason}"},
        ],
        "source":  source,
        "verdict": verdict,
    }

# ── Loaders ────────────────────────────────────────────────────────────────────

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
            lines_read = attacks = benign = 0
            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
                # Skip Zeek header comment lines
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                # Standard Zeek conn.log has 21 tab-separated fields.
                # The last tab field bundles "tunnel_parents label detailed-label"
                # as space-separated sub-tokens (IoT-23 specific formatting).
                if len(parts) < 21:
                    continue
                try:
                    proto      = parts[6]
                    duration   = parts[8]
                    orig_bytes = parts[9]
                    resp_bytes = parts[10]
                    conn_state = parts[11]
                    orig_pkts  = parts[16]
                    resp_pkts  = parts[18]
                    # Last field: "tunnel_parents   label   detailed-label"
                    # Label is always "Malicious" or "Benign" — search directly.
                    last = parts[-1]
                    if "Malicious" in last:
                        label = "Malicious"
                    elif "Benign" in last:
                        label = "Benign"
                    else:
                        continue
                except IndexError:
                    continue

                if label == "Malicious":
                    verdict = "ATTACK"
                elif label == "Benign":
                    verdict = "FALSE POSITIVE"
                else:
                    continue

                # skip "-" placeholders
                if orig_bytes == "-": orig_bytes = "0"
                if resp_bytes == "-": resp_bytes = "0"
                if duration   == "-": duration   = "0"
                if orig_pkts  == "-": orig_pkts  = "0"
                if resp_pkts  == "-": resp_pkts  = "0"

                bucket = samples[verdict]
                if len(bucket) < MAX_PER_SOURCE_CLASS:
                    bucket.append(make_sample(
                        proto, duration, orig_pkts, resp_pkts,
                        orig_bytes, resp_bytes, conn_state, verdict, "iot23"
                    ))
                lines_read += 1
                if verdict == "ATTACK":  attacks += 1
                else:                    benign  += 1

            print(f"    {member.name}: {attacks} attacks, {benign} benign")

    total = len(samples["ATTACK"]) + len(samples["FALSE POSITIVE"])
    print(f"  IoT-23 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]


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
                conn_state = row.get("State", "-").strip()
                tot_pkts   = row.get("TotPkts",  "0").strip()
                src_bytes  = row.get("SrcBytes",  "0").strip()
                tot_bytes  = row.get("TotBytes",  "0").strip()
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
                        src_bytes, dst_bytes, conn_state, verdict, "ctu13"
                    ))
                if verdict == "ATTACK":  attacks += 1
                else:                    benign  += 1

            print(f"    {member.name}: {attacks} attacks, {benign} benign")

    print(f"  CTU-13 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]


def load_unsw(dataset_dir):
    """Read UNSW-NB15 from CSV or parquet files."""
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] UNSW-NB15 directory not found: {dataset_dir}")
        return []

    # Try parquet first (Network-Flows subdir), then CSV
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
    samples = {"ATTACK": [], "FALSE POSITIVE": []}

    for fpath in files:
        print(f"  Reading {os.path.basename(fpath)} ...")
        try:
            if fpath.endswith(".parquet"):
                df = pd.read_parquet(fpath)
            else:
                df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        df.columns = [c.strip().lower() for c in df.columns]

        # UNSW-NB15 column names: proto, state, dur, sbytes, dbytes, spkts, dpkts, label, attack_cat
        proto_col  = next((c for c in ["proto", "protocol"]            if c in df.columns), None)
        dur_col    = next((c for c in ["dur", "duration"]             if c in df.columns), None)
        spkts_col  = next((c for c in ["spkts", "orig_pkts"]         if c in df.columns), None)
        dpkts_col  = next((c for c in ["dpkts", "resp_pkts"]         if c in df.columns), None)
        sbytes_col = next((c for c in ["sbytes", "orig_bytes"]       if c in df.columns), None)
        dbytes_col = next((c for c in ["dbytes", "resp_bytes"]       if c in df.columns), None)
        state_col  = next((c for c in ["state", "conn_state"]        if c in df.columns), None)
        label_col  = next((c for c in ["binary_label", "label"]      if c in df.columns), None)

        if label_col is None:
            print(f"    No label column found, skipping.")
            continue

        df = df.replace([float("inf"), float("-inf")], float("nan")).dropna(subset=[label_col])

        attacks = benign = 0
        for _, row in df.iterrows():
            lv = row[label_col]
            try:
                verdict = "ATTACK" if int(float(lv)) == 1 else "FALSE POSITIVE"
            except (ValueError, TypeError):
                verdict = "ATTACK" if str(lv).strip() not in ("0", "Normal", "BENIGN") else "FALSE POSITIVE"

            bucket = samples[verdict]
            if len(bucket) >= MAX_PER_SOURCE_CLASS:
                continue

            bucket.append(make_sample(
                proto      = str(row[proto_col])  if proto_col  else "unknown",
                duration   = str(row[dur_col])    if dur_col    else "0",
                orig_pkts  = str(row[spkts_col])  if spkts_col  else "0",
                resp_pkts  = str(row[dpkts_col])  if dpkts_col  else "0",
                orig_bytes = str(row[sbytes_col]) if sbytes_col else "0",
                resp_bytes = str(row[dbytes_col]) if dbytes_col else "0",
                conn_state = str(row[state_col])  if state_col  else "-",
                verdict    = verdict,
                source     = "unsw",
            ))
            if verdict == "ATTACK": attacks += 1
            else:                   benign  += 1

        print(f"    {attacks} attacks, {benign} benign")

    print(f"  UNSW-NB15 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]


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

    # CICFlowMeter → Zeek mapping
    COLMAP = {
        "Protocol":                   "proto",
        "Flow Duration":              "duration_us",   # microseconds — convert
        "Total Fwd Packets":          "orig_pkts",
        "Total Backward Packets":     "resp_pkts",
        "Total Length of Fwd Packets":"orig_bytes",
        "Total Length of Bwd Packets":"resp_bytes",
        "Label":                      "label",
    }

    for fpath in csv_files:
        print(f"  Reading {os.path.basename(fpath)} ...")
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        df.columns = df.columns.str.strip()
        df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()

        avail = {v: k for k, v in COLMAP.items() if k in df.columns}
        if "label" not in avail:
            continue

        file_counts = {"ATTACK": 0, "FALSE POSITIVE": 0}
        attacks = benign = 0
        for _, row in df.iterrows():
            raw_label = str(row[avail["label"]]).strip()
            verdict   = "FALSE POSITIVE" if raw_label == "BENIGN" else "ATTACK"
            if file_counts[verdict] >= PER_FILE_CAP:
                continue
            bucket    = samples[verdict]
            if len(bucket) >= MAX_PER_SOURCE_CLASS:
                continue

            proto     = str(int(float(row[avail["proto"]]))) if "proto" in avail else "unknown"
            dur_us    = row[avail["duration_us"]] if "duration_us" in avail else 0
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


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    all_samples = []
    all_samples += load_iot23(DATASETS["iot23"])
    all_samples += load_ctu13(DATASETS["ctu13"])
    all_samples += load_unsw(DATASETS["unsw"])
    all_samples += load_cicids(DATASETS["cicids"])

    attacks = [s for s in all_samples if s["verdict"] == "ATTACK"]
    benign  = [s for s in all_samples if s["verdict"] == "FALSE POSITIVE"]

    print(f"\nRaw pool: {len(attacks)} attacks, {len(benign)} benign")

    random.shuffle(attacks)
    random.shuffle(benign)
    attacks = attacks[:FINAL_ATTACK]
    benign  = benign[:FINAL_BENIGN]

    final = attacks + benign
    random.shuffle(final)

    # Strip internal keys before writing
    with open(OUTPUT_FILE, "w") as f:
        for s in final:
            out = {"messages": s["messages"]}
            f.write(json.dumps(out) + "\n")

    print(f"\n✅ {len(final)} samples written to {OUTPUT_FILE}")
    print(f"   Attacks: {len(attacks)}  |  Benign: {len(benign)}")

    # Source breakdown
    from collections import Counter
    sources = Counter(s["source"] for s in final)
    for src, n in sorted(sources.items()):
        print(f"   {src:12s}: {n:>7,}")
