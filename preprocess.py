import pandas as pd
import json
import os
import random
from pathlib import Path

# --- Config ---
CSV_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]
OUTPUT_FILE = "ids_dataset.jsonl"
MAX_SAMPLES_PER_FILE = 3000   # up from 2000 — more training signal
BENIGN_RATIO = 0.4            # 40% benign, 60% attacks

FEATURE_COLS = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Bwd Packet Length Max",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "FIN Flag Count",
    "Average Packet Size", "Avg Fwd Segment Size",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
]
LABEL_COL = "Label"

# ── System prompt — MUST match benchmark.py exactly ──────────────────────────
SYSTEM_PROMPT = (
    "You are a network security analyst. "
    "Always respond with VERDICT: <ATTACK or FALSE POSITIVE> on the first line, "
    "followed by REASON: <brief explanation>."
)

# ── Varied reason templates per class ─────────────────────────────────────────
# Multiple templates prevent the model from memorising phrases instead of
# learning to read the feature values.
BENIGN_REASONS = [
    "This flow matches normal background traffic patterns. No indicators of compromise detected.",
    "Traffic characteristics are consistent with routine network activity. No anomalies observed.",
    "Packet rates, sizes, and flag distributions fall within expected baselines for legitimate traffic.",
    "Flow duration and byte counts indicate standard application-layer communication. No threat detected.",
    "Connection behavior is typical of normal client-server interaction. No suspicious indicators.",
]

ATTACK_REASONS = {
    "DDoS": [
        "Distributed Denial of Service attack detected. Abnormally high packet/byte rates from multiple sources.",
        "DDoS pattern identified. Excessive flow volume and packet rate consistent with volumetric flooding.",
        "Traffic surge detected with characteristics of a distributed denial of service attack.",
    ],
    "PortScan": [
        "Port scanning activity detected. Sequential connection attempts across multiple destination ports.",
        "Reconnaissance behavior identified. Rapid SYN packets probing multiple ports on the target.",
        "Network scanning pattern detected. Short-lived connections across a wide port range.",
    ],
    "Bot": [
        "Botnet communication pattern detected. Periodic beaconing behavior with C2 server characteristics.",
        "Automated bot traffic identified. Regular interval connections consistent with command-and-control activity.",
        "Botnet activity detected. Uniform packet timing and payload sizes suggest automated communication.",
    ],
    "Infilteration": [
        "Network infiltration attempt detected. Unusual inbound connection pattern bypassing perimeter controls.",
        "Infiltration activity identified. Abnormal data exfiltration pattern with high outbound byte volume.",
        "Lateral movement or infiltration detected. Connection pattern inconsistent with authorized access.",
    ],
    "Web Attack": [
        "Web-based attack detected. Anomalous HTTP request patterns consistent with exploitation attempts.",
        "Web application attack identified. Unusual request sizes and frequencies targeting application layer.",
        "Malicious web traffic detected. Request patterns consistent with brute-force or injection attacks.",
    ],
    "Heartbleed": [
        "Heartbleed (CVE-2014-0160) exploit attempt detected. Malformed TLS heartbeat request.",
        "TLS Heartbleed vulnerability exploit detected. Abnormal heartbeat payload sizes.",
    ],
    "DoS": [
        "Denial of Service attack detected. Flood of packets overwhelming target service.",
        "DoS attack identified. Extreme packet rate and connection volume targeting a single service.",
        "Service disruption attempt detected. Sustained high-volume traffic aimed at exhausting resources.",
    ],
    "FTP-Patator": [
        "FTP brute-force attack detected. High volume of failed authentication attempts.",
        "FTP credential stuffing identified. Rapid sequential login attempts on FTP service.",
        "Brute-force attack on FTP detected. Many short-lived connections with authentication failures.",
    ],
    "SSH-Patator": [
        "SSH brute-force attack detected. Repeated failed login attempts on port 22.",
        "SSH credential attack identified. High-frequency authentication attempts against SSH service.",
        "Brute-force SSH attack detected. Sequential login attempts with varied credentials.",
    ],
}


def label_to_verdict(label: str, rng: random.Random) -> tuple[str, str]:
    label = label.strip()
    if label == "BENIGN":
        return "FALSE POSITIVE", rng.choice(BENIGN_REASONS)
    for key, reasons in ATTACK_REASONS.items():
        if key.lower() in label.lower():
            return "ATTACK", rng.choice(reasons)
    return "ATTACK", f"Malicious activity detected. Traffic classified as: {label}."


def row_to_prompt(row: pd.Series, available_cols: list) -> str:
    lines = ["Analyze the following network flow and classify it as ATTACK or FALSE POSITIVE.\n"]
    for col in available_cols:
        val = row[col]
        lines.append(f"  {col.strip()}: {val}")
    return "\n".join(lines)


def process_files(file_list: list) -> list:
    all_samples = []
    rng = random.Random(42)

    for fpath in file_list:
        if not os.path.exists(fpath):
            print(f"[SKIP] File not found: {fpath}")
            continue

        print(f"[LOAD] {fpath}")
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()

        label_col = "Label"
        if label_col not in df.columns:
            print(f"  [WARN] No 'Label' column found. Columns: {list(df.columns[:5])}")
            continue

        df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()
        avail = [c for c in FEATURE_COLS if c.strip() in df.columns]

        benign = df[df[label_col] == "BENIGN"]
        attacks = df[df[label_col] != "BENIGN"]

        n_benign = min(int(MAX_SAMPLES_PER_FILE * BENIGN_RATIO), len(benign))
        n_attack = min(MAX_SAMPLES_PER_FILE - n_benign, len(attacks))

        sampled = pd.concat([
            benign.sample(n_benign, random_state=42),
            attacks.sample(n_attack, random_state=42),
        ]).sample(frac=1, random_state=42)

        print(f"  Sampled {n_benign} benign + {n_attack} attacks from {len(df)} rows")

        for _, row in sampled.iterrows():
            verdict, reason = label_to_verdict(row[label_col], rng)
            prompt = row_to_prompt(row, avail)
            response = f"VERDICT: {verdict}\nREASON: {reason}"
            all_samples.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            })

    return all_samples


if __name__ == "__main__":
    samples = process_files(CSV_FILES)
    with open(OUTPUT_FILE, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"\n✅ Saved {len(samples)} samples to {OUTPUT_FILE}")
