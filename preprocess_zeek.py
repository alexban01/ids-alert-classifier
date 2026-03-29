"""
preprocess_zeek.py — Build training JSONL from Zeek-native / Zeek-compatible datasets.

Datasets supported:
  - IoT-23        : native Zeek conn.log.labeled files
  - CTU-13        : binetflow (Argus bidirectional flows)
  - UNSW-NB15     : CSV/parquet (Bro/Zeek-generated, good column overlap)
  - CICIDS2017    : CICFlowMeter CSVs (already present) — mapped to Zeek schema
  - UWF-ZeekData24: real Zeek conn.log from university cyber range (MITRE labeled)
  - CTU-Normal    : benign-only Zeek conn.log from real user browsing

Outputs:
  zeek_dataset.jsonl       — training samples (~90% per source)
  zeek_dataset_eval.jsonl  — held-out eval samples (~10% per source, stratified)

Zeek conn.log feature set used across all sources:
  proto, service, duration, orig_pkts, resp_pkts, orig_bytes, resp_bytes,
  conn_state, bytes_per_sec, orig_bytes_per_pkt, resp_bytes_per_pkt
"""

import os
import json
import random
import tarfile
import glob
import math

import pandas as pd

from prompt_utils import SYSTEM_PROMPT, build_prompt

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_FILE    = "zeek_dataset.jsonl"
EVAL_FILE     = "zeek_dataset_eval.jsonl"
RANDOM_SEED   = 42
EVAL_FRAC     = 0.10            # fraction of each source held out for eval

# This is for fast training on my 3070
TRAINING_FACTOR = 0.1

MAX_PER_SOURCE_CLASS = int(80_000 * TRAINING_FACTOR)   # default cap per (source, class)
IOT23_BENIGN_CAP     = int(20_000 * TRAINING_FACTOR)   # IoT-23 benign is 89% S0-dominated; reduce to avoid
                                   # "S0 = benign" bias that hurts real-world SF traffic
CTU_NORMAL_CAP       = int(100_000 * TRAINING_FACTOR)  # increase — only significant SF benign source

# v7: 2:1 benign:attack ratio — real networks are overwhelmingly benign,
# training balanced (1:1) makes the model trigger-happy on real traffic.
FINAL_BENIGN  = int(240_000 * TRAINING_FACTOR)
FINAL_ATTACK  = int(120_000 * TRAINING_FACTOR)

DATASETS = {
    "iot23":      "datasets/iot-23/iot_23_datasets_small.tar.gz",
    "ctu13":      "datasets/ctu-13/CTU-13-Dataset.tar.bz2",
    "unsw":       "datasets/unsw-nb15/",
    "cicids":     ".",   # looks for *.pcap_ISCX.csv in cwd
    "uwf":        "datasets/uwf-zeekdata24/",
    "ctu_normal": "datasets/ctu-normal/",
}

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

# ── Sample builder ─────────────────────────────────────────────────────────────
def make_sample(proto, duration, orig_pkts, resp_pkts,
                orig_bytes, resp_bytes, conn_state, verdict, source, service="-"):
    prompt  = build_prompt(proto, duration, orig_pkts, resp_pkts,
                           orig_bytes, resp_bytes, conn_state, service)
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
                    service    = parts[7]
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

                cap    = IOT23_BENIGN_CAP if verdict == "FALSE POSITIVE" else MAX_PER_SOURCE_CLASS
                bucket = samples[verdict]
                if len(bucket) < cap:
                    bucket.append(make_sample(
                        proto, duration, orig_pkts, resp_pkts,
                        orig_bytes, resp_bytes, conn_state, verdict, "iot23",
                        service=service,
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
    """Read CTU-13 binetflow files from tar.bz2 archive.

    Binetflow uses Argus state notation — mapped to Zeek conn_state equivalents
    so the model only ever sees Zeek states during training.
    """
    # Argus binetflow state → Zeek conn_state
    CTU_STATE_MAP = {
        "INT":        "S1",    # mid-flow established, no FIN seen
        "CON":        "SF",    # completed connection
        "FIN":        "SF",    # completed with FIN
        "FSPA_FSPA":  "SF",    # FIN bidirectional = completed
        "FSA_FSA":    "SF",
        "SPA_FSPA":   "SF",
        "PA_PA":      "OTH",   # PSH-ACK only, no SYN seen
        "EST":        "S1",    # established
        "S_":         "S0",    # SYN only, no response
        "REQ":        "S0",
        "SRPA_SPA":   "RSTO",  # RST from originator
        "SRST":       "RSTO",
    }

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
                conn_state = CTU_STATE_MAP.get(raw_state, "-")
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
                        src_bytes, dst_bytes, conn_state, verdict, "ctu13",
                        service="-",  # binetflow has no app-layer service field
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
        svc_col    = next((c for c in ["service"]                    if c in df.columns), None)
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
                service    = str(row[svc_col]).strip() if svc_col else "-",
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

    # v7: UWF attacks are 100% "Credential Access" (short SF TCP, ~0.02s) —
    # indistinguishable from normal web connections at the flow level (2% recall).
    # Training on them teaches "short SF TCP = ATTACK", causing false positives on
    # legitimate traffic. Use UWF for benign diversity only.
    print(f"[UWF-ZeekData24] Loading {len(csv_files)} CSV(s) from {dataset_dir} (benign only)")
    samples = {"ATTACK": [], "FALSE POSITIVE": []}

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

        attacks = benign = 0
        for _, row in df.iterrows():
            if label_col == "label_binary":
                verdict = "ATTACK" if str(row[label_col]).strip() == "True" else "FALSE POSITIVE"
            else:
                verdict = "ATTACK" if str(row[label_col]).strip() != "none" else "FALSE POSITIVE"

            # Skip attack samples — unlearnable from flow features, harmful to train on
            if verdict == "ATTACK":
                continue

            bucket = samples[verdict]
            if len(bucket) >= MAX_PER_SOURCE_CLASS:
                continue

            # UWF uses empty strings for missing values — pass through as-is
            # (build_prompt / _safe will convert to N/A)
            proto      = str(row.get("proto", "unknown")).strip()
            service    = str(row.get("service", "-")).strip()
            duration   = str(row.get("duration", "")).strip()
            orig_pkts  = str(row.get("orig_pkts", "")).strip()
            resp_pkts  = str(row.get("resp_pkts", "")).strip()
            orig_bytes = str(row.get("orig_bytes", "")).strip()
            resp_bytes = str(row.get("resp_bytes", "")).strip()
            conn_state = str(row.get("conn_state", "-")).strip()

            # pandas converts empty CSV cells to nan
            if service    in ("nan", "None"): service    = "-"
            if duration   in ("nan", "None"): duration   = ""
            if orig_pkts  in ("nan", "None"): orig_pkts  = ""
            if resp_pkts  in ("nan", "None"): resp_pkts  = ""
            if orig_bytes in ("nan", "None"): orig_bytes = ""
            if resp_bytes in ("nan", "None"): resp_bytes = ""

            bucket.append(make_sample(
                proto, duration, orig_pkts, resp_pkts,
                orig_bytes, resp_bytes, conn_state, verdict, "uwf",
                service=service,
            ))
            if verdict == "ATTACK": attacks += 1
            else:                   benign  += 1

        print(f"    {attacks} attacks, {benign} benign")

    print(f"  UWF-ZeekData24 total: {len(samples['ATTACK'])} attacks, "
          f"{len(samples['FALSE POSITIVE'])} benign")
    return samples["ATTACK"] + samples["FALSE POSITIVE"]


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
    total = 0

    for fpath in log_files:
        count = 0
        with open(fpath) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 21:
                    continue

                if len(samples) >= CTU_NORMAL_CAP:
                    break

                proto      = parts[6]
                service    = parts[7]
                duration   = parts[8]
                orig_bytes = parts[9]
                resp_bytes = parts[10]
                conn_state = parts[11]
                orig_pkts  = parts[16]
                resp_pkts  = parts[18]

                # All CTU-Normal traffic is benign — pass - values through as-is
                samples.append(make_sample(
                    proto, duration, orig_pkts, resp_pkts,
                    orig_bytes, resp_bytes, conn_state,
                    "FALSE POSITIVE", "ctu_normal",
                    service=service,
                ))
                count += 1

        total += count
        print(f"  {os.path.basename(fpath)}: {count} benign entries")

    print(f"  CTU-Normal total: {len(samples)} benign")
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from collections import Counter, defaultdict
    random.seed(RANDOM_SEED)

    all_samples = []
    all_samples += load_iot23(DATASETS["iot23"])
    all_samples += load_ctu13(DATASETS["ctu13"])
    all_samples += load_unsw(DATASETS["unsw"])
    # CICIDS2017 dropped in v7: CICFlowMeter produces proto="unknown" and
    # conn_state="-" for every flow — both fields are always missing, so the
    # model can't learn Zeek-discriminative patterns from this source.
    # Also has documented label quality issues (~10-15% mislabeled).
    # all_samples += load_cicids(DATASETS["cicids"])
    all_samples += load_uwf(DATASETS["uwf"])
    all_samples += load_ctu_normal(DATASETS["ctu_normal"])

    # ── Source-stratified train/eval split ────────────────────────────────────
    # Hold out EVAL_FRAC from each (source, verdict) bucket so eval distribution
    # matches the real-world variety of all sources — not just whatever ends up
    # in a random 10% slice of the merged pool.
    by_bucket = defaultdict(list)
    for s in all_samples:
        by_bucket[(s["source"], s["verdict"])].append(s)

    train_pool, eval_pool = [], []
    for bucket_samples in by_bucket.values():
        random.shuffle(bucket_samples)
        n_eval = max(1, int(len(bucket_samples) * EVAL_FRAC))
        eval_pool.extend(bucket_samples[:n_eval])
        train_pool.extend(bucket_samples[n_eval:])

    # ── Subsample train pool to target ratio ──────────────────────────────────
    attacks = [s for s in train_pool if s["verdict"] == "ATTACK"]
    benign  = [s for s in train_pool if s["verdict"] == "FALSE POSITIVE"]

    print(f"\nRaw train pool: {len(attacks)} attacks, {len(benign)} benign")
    print(f"Eval pool     : {sum(1 for s in eval_pool if s['verdict']=='ATTACK')} attacks, "
          f"{sum(1 for s in eval_pool if s['verdict']=='FALSE POSITIVE')} benign")

    random.shuffle(attacks)
    random.shuffle(benign)
    attacks = attacks[:FINAL_ATTACK]
    benign  = benign[:FINAL_BENIGN]

    final_train = attacks + benign
    random.shuffle(final_train)
    random.shuffle(eval_pool)

    def write_jsonl(path, samples):
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps({"messages": s["messages"]}) + "\n")

    write_jsonl(TRAIN_FILE, final_train)
    write_jsonl(EVAL_FILE,  eval_pool)

    print(f"\n✅ {len(final_train)} train samples → {TRAIN_FILE}")
    print(f"   Attacks: {len(attacks):>7,}  |  Benign: {len(benign):>7,}  "
          f"(ratio 1:{len(benign)/max(len(attacks),1):.1f})")
    print(f"✅ {len(eval_pool)} eval samples  → {EVAL_FILE}")

    print(f"\n   Train source breakdown:")
    sources = Counter(s["source"] for s in final_train)
    for src, n in sorted(sources.items()):
        a = sum(1 for s in final_train if s["source"] == src and s["verdict"] == "ATTACK")
        b = sum(1 for s in final_train if s["source"] == src and s["verdict"] == "FALSE POSITIVE")
        print(f"   {src:12s}: {n:>7,}  (atk {a:>6,} / ben {b:>6,})")
