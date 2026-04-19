"""
bench_loaders.py — data loaders for benchmark_realworld.py.

Each loader returns a flat list of sample dicts (built with make_sample).
All loaders are self-contained: no ML imports, no GPU, safe to run in a
ProcessPoolExecutor worker.

Loaders:
    load_iot23(archive_path)      — IoT-23 conn.log.labeled from tar.gz
    load_ctu13(dataset_dir)       — CTU-13 binetflow CSVs from extracted directory
    load_uwf(dataset_dir)         — UWF-ZeekData24 Zeek conn.log CSVs
    load_ctu_normal(dataset_dir)  — CTU-Normal benign-only Zeek conn.log
    load_ctu_sme11()              — CTU-SME-11 Amazon Echo OOD probe (easy OOD —
                                    inbound IoT scan traffic; downloads from Zenodo)
    load_ctu_win7ad()             — CTU-SME-11 Windows7AD-1 primary OOD probe
                                    (infected Windows VM, outbound lateral movement +
                                    Trickbot C2; downloads from Zenodo if not cached)
    load_ctu_botnet3()            — CTU-Malware Botnet-3 (Kelihos) hard floor OOD probe
                                    (P2P spam botnet; attacks only; local conn.log)
"""

import os
import sys
import re
import tarfile
import glob
import urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from prompt_utils import build_prompt

# ── Constants ───────────────────────────────────────────────────────────────────
CAP = 300 # max samples per (source, class)

# CTU-SME-11 archives — Zenodo record 7958259
# Echo (RETIRED): inbound internet background scanning, invalid OOD ground truth
# Windows7AD-1 (PRIMARY OOD): infected Windows VM, outbound lateral movement + Trickbot C2
CTU_SME11_ARCHIVE    = "CTU-SME-11_Honeypot-Assistant-Amazon-Echo1stGen-1_v1.0.0.tar.bz2"
CTU_SME11_URL        = f"https://zenodo.org/records/7958259/files/{CTU_SME11_ARCHIVE}?download=1"
CTU_WIN7AD_ARCHIVE   = "CTU-SME-11_Experiment-VM-Microsoft-Windows7AD-1_v1.0.0.tar.bz2"
CTU_WIN7AD_LOCAL     = "CTU-SME-11_Win7AD.tar.bz2"   # shorter local filename
CTU_WIN7AD_URL       = f"https://zenodo.org/records/7958259/files/{CTU_WIN7AD_ARCHIVE}?download=1"
CTU_CAPTURE_DIR      = "test_captures"
CTU_BOTNET3_LOG      = "test_captures/CTU-Malware-Capture-Botnet-3_conn.log"


# ── Sample helper ────────────────────────────────────────────────────────────────
def make_sample(proto, duration, orig_pkts, resp_pkts,
                orig_bytes, resp_bytes, conn_state,
                ground_truth, source, raw_label, service="-",
                resp_port="-", orig_port="-", ts=None,
                orig_h=None, resp_h=None, uid=None, group_id=None):
    return {
        "prompt":       build_prompt(proto, duration, orig_pkts, resp_pkts,
                                     orig_bytes, resp_bytes, conn_state, service,
                                     resp_port=resp_port, orig_port=orig_port),
        "ground_truth": ground_truth,
        "source":       source,
        "raw_label":    raw_label,
        "ts":           ts,
        "uid":          uid,
        "orig_h":       orig_h,
        "orig_p":       orig_port,
        "resp_h":       resp_h,
        "resp_p":       resp_port,
        "proto":        proto,
        "service":      service,
        "duration":     duration,
        "orig_pkts":    orig_pkts,
        "resp_pkts":    resp_pkts,
        "orig_bytes":   orig_bytes,
        "resp_bytes":   resp_bytes,
        "conn_state":   conn_state,
        "group_id":     group_id or source,
    }


# ── Standard loaders ────────────────────────────────────────────────────────────

def load_iot23(archive_path):
    """Native Zeek conn.log.labeled from IoT-23 tar.gz or extracted directory.

    Accepts either:
      - a .tar.gz archive path (streams directly)
      - a directory path (walks for conn.log.labeled files)

    Last tab field bundles: tunnel_parents label detailed-label (space-separated).
    Detailed label examples: C&C, DDoS, Okiru, PartOfAHorizontalPortScan, etc.
    """
    # Resolve: if archive_path is a directory, walk it; otherwise open as tar.gz.
    # Also check for extracted sibling directory (strip .tar.gz suffix).
    extracted_dir = None
    if os.path.isdir(archive_path):
        extracted_dir = archive_path
    elif archive_path.endswith(".tar.gz"):
        candidate = archive_path[:-7]  # strip .tar.gz
        if os.path.isdir(candidate):
            extracted_dir = candidate
        # Also check parent dir if archive basename stripped gives a subdirectory
        parent = os.path.dirname(archive_path)
        if extracted_dir is None and os.path.isdir(parent):
            # Walk parent for conn.log.labeled — covers the iot-23/ extracted layout
            found_files = glob.glob(os.path.join(parent, "**", "conn.log.labeled"), recursive=True)
            if found_files:
                extracted_dir = parent

    if extracted_dir is None and not os.path.isfile(archive_path):
        print(f"[SKIP] IoT-23 not found: {archive_path}")
        return []

    buckets = defaultdict(list)

    def _process_lines(lines, file_label):
        for raw in lines:
            if isinstance(raw, bytes):
                line = raw.decode("utf-8", errors="replace").strip()
            else:
                line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 21:
                continue

            last = parts[-1]
            if "Malicious" in last:
                verdict = "ATTACK"
            elif "Benign" in last:
                verdict = "FALSE POSITIVE"
            else:
                continue

            if len(buckets[verdict]) >= CAP:
                continue

            sub = last.split()
            detailed = sub[2] if len(sub) >= 3 else sub[-1] if sub else "-"
            raw_label = detailed if verdict == "ATTACK" else "Benign"

            try:
                buckets[verdict].append(make_sample(
                    proto      = parts[6],
                    duration   = parts[8],
                    orig_pkts  = parts[16],
                    resp_pkts  = parts[18],
                    orig_bytes = parts[9],
                    resp_bytes = parts[10],
                    conn_state = parts[11],
                    ground_truth = verdict,
                    source     = "iot23",
                    raw_label  = raw_label,
                    service    = parts[7],
                    orig_port  = parts[3],
                    resp_port  = parts[5],
                    ts         = parts[0],
                    uid        = parts[1],
                    orig_h     = parts[2],
                    resp_h     = parts[4],
                    group_id   = file_label,
                ))
            except IndexError:
                continue

            if all(len(v) >= CAP for v in buckets.values()):
                return True  # signal early stop
        return False

    if extracted_dir is not None:
        log_files = glob.glob(os.path.join(extracted_dir, "**", "conn.log.labeled"), recursive=True)
        print(f"[IoT-23] Found {len(log_files)} conn.log.labeled file(s) in {extracted_dir}")
        for path in log_files:
            with open(path, "r", errors="replace") as f:
                done = _process_lines(f, path)
            if done:
                break
    else:
        print(f"[IoT-23] Opening archive {archive_path} ...")
        with tarfile.open(archive_path, "r:gz") as tf:
            members = [m for m in tf.getmembers()
                       if m.name.endswith("conn.log.labeled") and m.isfile()]
            print(f"  {len(members)} conn.log.labeled file(s)")
            for member in members:
                f = tf.extractfile(member)
                if f is None:
                    continue
                done = _process_lines(f, member.name)
                if done:
                    break

    atk = len(buckets["ATTACK"])
    ben = len(buckets["FALSE POSITIVE"])
    print(f"  IoT-23: {atk} attacks, {ben} benign")
    return buckets["ATTACK"] + buckets["FALSE POSITIVE"]


def load_ctu13(dataset_dir):
    """CTU-13 binetflow CSVs from extracted dataset directory.

    Has TotPkts only (split 50/50 between orig/resp).
    Label: contains 'Botnet' → ATTACK, 'Normal' → FP, 'Background' → skip.
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

    print(f"[CTU-13] Found {len(binetflow_files)} binetflow file(s) in {dataset_dir}")
    buckets = defaultdict(list)

    for filepath in binetflow_files:
        if all(len(v) >= CAP for v in [buckets["ATTACK"], buckets["FALSE POSITIVE"]]):
            break
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
                    continue

                if len(buckets[verdict]) >= CAP:
                    continue

                tot_pkts  = row.get("TotPkts",  "0").strip()
                src_bytes = row.get("SrcBytes",  "0").strip()
                tot_bytes = row.get("TotBytes",  "0").strip()
                try:
                    half      = str(int(float(tot_pkts)) // 2)
                    dst_bytes = str(max(0.0, float(tot_bytes) - float(src_bytes)))
                except ValueError:
                    half = "0"; dst_bytes = "0"

                buckets[verdict].append(make_sample(
                    proto      = row.get("Proto", "unknown").strip().lower(),
                    duration   = row.get("Dur", "0").strip(),
                    orig_pkts  = half,
                    resp_pkts  = half,
                    orig_bytes = src_bytes,
                    resp_bytes = dst_bytes,
                    conn_state = row.get("State", "-").strip(),
                    ground_truth = verdict,
                    source     = "ctu13",
                    raw_label  = label.strip(),
                    service    = "-",  # binetflow has no app-layer service field
                    orig_port  = row.get("Sport", "-").strip(),
                    resp_port  = row.get("Dport", "-").strip(),
                    orig_h     = row.get("SrcAddr", "").strip(),
                    resp_h     = row.get("DstAddr", "").strip(),
                    group_id   = filepath,
                ))

    atk = len(buckets["ATTACK"])
    ben = len(buckets["FALSE POSITIVE"])
    print(f"  CTU-13: {atk} attacks, {ben} benign")
    return buckets["ATTACK"] + buckets["FALSE POSITIVE"]


def load_uwf(dataset_dir):
    """UWF-ZeekData24: real Zeek conn.log from UWF cyber range (MITRE-labeled).

    label_binary: "True" → ATTACK, "False" → FP
    label_tactic: MITRE tactic name (e.g. lateral_movement, command_and_control, none)
    """
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] UWF-ZeekData24 not found: {dataset_dir}")
        return []

    csv_files = sorted(glob.glob(os.path.join(dataset_dir, "**/*.csv"), recursive=True))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith(".")]
    if not csv_files:
        print(f"[SKIP] No CSVs in {dataset_dir}")
        return []

    print(f"[UWF-ZeekData24] {len(csv_files)} CSV(s) from {dataset_dir}")
    buckets = defaultdict(list)

    for fpath in csv_files:
        if all(len(buckets[v]) >= CAP for v in ["ATTACK", "FALSE POSITIVE"]):
            break
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"  ERROR {fpath}: {e}")
            continue
        df.columns = [c.strip() for c in df.columns]

        label_bin    = next((c for c in ["label_binary"] if c in df.columns), None)
        label_tactic = next((c for c in ["label_tactic"] if c in df.columns), None)
        if label_bin is None:
            continue

        for _, row in df.iterrows():
            verdict = ("ATTACK" if str(row[label_bin]).strip() == "True"
                       else "FALSE POSITIVE")
            if len(buckets[verdict]) >= CAP:
                continue

            if label_tactic:
                raw = str(row[label_tactic]).strip()
                raw_label = raw if raw and raw != "none" else (
                    "Benign" if verdict == "FALSE POSITIVE" else "unknown_attack"
                )
            else:
                raw_label = "ATTACK" if verdict == "ATTACK" else "Benign"

            def _clean(val):
                s = str(val).strip()
                return "" if s in ("nan", "None", "NaN") else s

            svc = _clean(row.get("service", "-")) or "-"
            buckets[verdict].append(make_sample(
                proto      = _clean(row.get("proto", "unknown")),
                duration   = _clean(row.get("duration", "")),
                orig_pkts  = _clean(row.get("orig_pkts", "")),
                resp_pkts  = _clean(row.get("resp_pkts", "")),
                orig_bytes = _clean(row.get("orig_bytes", "")),
                resp_bytes = _clean(row.get("resp_bytes", "")),
                conn_state = _clean(row.get("conn_state", "-")) or "-",
                ground_truth = verdict,
                source     = "uwf",
                raw_label  = raw_label,
                service    = svc,
                orig_port  = _clean(row.get("id.orig_p", row.get("orig_p", row.get("src_port_zeek", "-")))) or "-",
                resp_port  = _clean(row.get("id.resp_p", row.get("resp_p", row.get("dest_port_zeek", "-")))) or "-",
                ts         = _clean(row.get("ts", row.get("timestamp", ""))) or None,
                uid        = _clean(row.get("uid", "")) or None,
                orig_h     = _clean(row.get("id.orig_h", row.get("orig_h", row.get("src_ip", "")))) or None,
                resp_h     = _clean(row.get("id.resp_h", row.get("resp_h", row.get("dest_ip", "")))) or None,
                group_id   = os.path.basename(fpath),
            ))

    atk = len(buckets["ATTACK"])
    ben = len(buckets["FALSE POSITIVE"])
    print(f"  UWF: {atk} attacks, {ben} benign")
    return buckets["ATTACK"] + buckets["FALSE POSITIVE"]


def load_ctu_normal(dataset_dir):
    """CTU-Normal: benign-only Zeek conn.log TSV (standard 21-field format)."""
    if not os.path.isdir(dataset_dir):
        print(f"[SKIP] CTU-Normal not found: {dataset_dir}")
        return []

    log_files = sorted(glob.glob(os.path.join(dataset_dir, "*.log")))
    if not log_files:
        print(f"[SKIP] No .log files in {dataset_dir}")
        return []

    print(f"[CTU-Normal] {len(log_files)} conn.log file(s) from {dataset_dir}")
    samples = []

    for fpath in log_files:
        if len(samples) >= CAP:
            break
        with open(fpath) as f:
            for line in f:
                if len(samples) >= CAP:
                    break
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 21:
                    continue
                samples.append(make_sample(
                    proto      = parts[6],
                    duration   = parts[8],
                    orig_pkts  = parts[16],
                    resp_pkts  = parts[18],
                    orig_bytes = parts[9],
                    resp_bytes = parts[10],
                    conn_state = parts[11],
                    ground_truth = "FALSE POSITIVE",
                    source     = "ctu_normal",
                    raw_label  = "Benign",
                    service    = parts[7],
                    orig_port  = parts[3],
                    resp_port  = parts[5],
                    ts         = parts[0],
                    uid        = parts[1],
                    orig_h     = parts[2],
                    resp_h     = parts[4],
                    group_id   = os.path.basename(fpath),
                ))

    print(f"  CTU-Normal: 0 attacks, {len(samples)} benign")
    return samples


# ── CTU-SME-11 OOD loader ───────────────────────────────────────────────────────

def _bench_download(url, local_path):
    """Download url to local_path if not already cached."""
    if os.path.isfile(local_path):
        print(f"  [cache] {os.path.basename(local_path)}")
        return local_path
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    print(f"  Downloading {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(local_path, "wb") as f:
                f.write(resp.read())
        print(f"    {os.path.getsize(local_path) // 1024} KB → {local_path}")
        return local_path
    except Exception as e:
        if os.path.isfile(local_path):
            os.remove(local_path)
        print(f"  [ERROR] Download failed: {e}")
        return None


def load_ctu_sme11():
    """Load CTU-SME-11 Amazon Echo honeypot capture as easy OOD eval source.

    CTU-SME-11 is a 7-day enterprise network capture from Stratosphere Lab
    (Zenodo record 7958259). The Amazon Echo device has 76k flows (~48% malicious)
    and is the smallest archive (742.9 MB). File format is Zeek conn.log.labeled —
    identical to IoT-23 (same lab), with 'Malicious'/'Benign'/'Background' labels
    in the last tab-separated field. Never included in training data.
    Role: easy OOD — inbound IoT scan traffic with some structural similarity to
    IoT-23 training data. LLM v9.1 scores +0.373 MCC here (partial transfer).
    """
    print(f"\n[CTU-SME-11 OOD / Amazon Echo]")
    os.makedirs(CTU_CAPTURE_DIR, exist_ok=True)

    local_path   = os.path.join(CTU_CAPTURE_DIR, CTU_SME11_ARCHIVE)
    archive_path = _bench_download(CTU_SME11_URL, local_path)
    if archive_path is None:
        print("  [SKIP] Archive download failed")
        return []

    print(f"  Opening {os.path.basename(archive_path)} ...")
    buckets = defaultdict(list)

    try:
        with tarfile.open(archive_path, "r:bz2") as tf:
            members = [m for m in tf.getmembers()
                       if m.name.endswith("conn.log.labeled") and m.isfile()]
            print(f"  {len(members)} conn.log.labeled file(s)")

            for member in members:
                if all(len(buckets[v]) >= CAP for v in ["ATTACK", "FALSE POSITIVE"]):
                    break
                f = tf.extractfile(member)
                if f is None:
                    continue
                for raw in f:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 23:
                        continue

                    # CTU-SME-11: label and detailedlabel are separate tab columns
                    # (unlike IoT-23 where they were bundled into the last field)
                    label_col    = parts[21]   # "Malicious" / "Benign" / "Background"
                    detailed_col = parts[22]   # e.g. "From_malicious-To_benign-Discovery"

                    if label_col == "Malicious":
                        verdict = "ATTACK"
                    elif label_col == "Benign":
                        verdict = "FALSE POSITIVE"
                    else:
                        continue  # Background → skip

                    if len(buckets[verdict]) >= CAP:
                        continue

                    raw_label = detailed_col if verdict == "ATTACK" else "Benign"

                    try:
                        buckets[verdict].append(make_sample(
                            proto        = parts[6],
                            duration     = parts[8],
                            orig_pkts    = parts[16],
                            resp_pkts    = parts[18],
                            orig_bytes   = parts[9],
                            resp_bytes   = parts[10],
                            conn_state   = parts[11],
                            ground_truth = verdict,
                            source       = "ctu_sme11",
                            raw_label    = raw_label,
                            service      = parts[7],
                            orig_port    = parts[3],
                            resp_port    = parts[5],
                            ts           = parts[0],
                            uid          = parts[1],
                            orig_h       = parts[2],
                            resp_h       = parts[4],
                            group_id     = member.name,
                        ))
                    except IndexError:
                        continue

                    if all(len(v) >= CAP for v in buckets.values()):
                        break
    except Exception as e:
        print(f"  [SKIP] Archive parse error: {e}")
        return []

    atk = len(buckets["ATTACK"])
    ben = len(buckets["FALSE POSITIVE"])
    print(f"  CTU-SME-11 (OOD): {atk} attacks, {ben} benign sampled")
    return buckets["ATTACK"] + buckets["FALSE POSITIVE"]


def load_ctu_win7ad():
    """Load CTU-SME-11 Windows7AD-1 as primary OOD eval source (V10.0+).

    An infected Windows 7 Active Directory VM from Stratosphere Lab's CTU-SME-11
    dataset (Zenodo record 7958259). 7-day enterprise capture, 184k flows (1.1%
    malicious). All malicious flows originate FROM the internal VM (192.168.x.x
    as orig_h) — clean outbound attack ground truth.

    Attack mix (2,038 malicious flows total):
      89% Human attacks — REJ/RSTR to RDP/3389, Kerberos/88/464, SMB/445, MSRPC
      11% Trickbot C2   — RSTO to non-standard ports 4134 and 22299

    File format: Zeek conn.log.labeled with separate label (col 21) and
    detailedlabel (col 22) columns — same as the Echo capture.
    Local cache: test_captures/CTU-SME-11_Win7AD.tar.bz2
    """
    print(f"\n[CTU-SME-11 OOD / Windows7AD-1]")
    os.makedirs(CTU_CAPTURE_DIR, exist_ok=True)

    local_path   = os.path.join(CTU_CAPTURE_DIR, CTU_WIN7AD_LOCAL)
    archive_path = _bench_download(CTU_WIN7AD_URL, local_path)
    if archive_path is None:
        print("  [SKIP] Archive download failed")
        return []

    print(f"  Opening {os.path.basename(archive_path)} ...")
    buckets = defaultdict(list)

    try:
        with tarfile.open(archive_path, "r:bz2") as tf:
            members = [m for m in tf.getmembers()
                       if m.name.endswith("conn.log.labeled") and m.isfile()]
            print(f"  {len(members)} conn.log.labeled file(s)")

            for member in members:
                if all(len(buckets[v]) >= CAP for v in ["ATTACK", "FALSE POSITIVE"]):
                    break
                f = tf.extractfile(member)
                if f is None:
                    continue
                for raw in f:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 23:
                        continue

                    label_col   = parts[21]   # "Malicious" / "Benign" / "Background"
                    detailed_col = parts[22]  # e.g. "From_malicious-To_benign-Human_attacks"

                    if label_col == "Malicious":
                        verdict = "ATTACK"
                    elif label_col == "Benign":
                        verdict = "FALSE POSITIVE"
                    else:
                        continue  # Background → skip

                    if len(buckets[verdict]) >= CAP:
                        continue

                    raw_label = detailed_col if verdict == "ATTACK" else "Benign"

                    try:
                        buckets[verdict].append(make_sample(
                            proto        = parts[6],
                            duration     = parts[8],
                            orig_pkts    = parts[16],
                            resp_pkts    = parts[18],
                            orig_bytes   = parts[9],
                            resp_bytes   = parts[10],
                            conn_state   = parts[11],
                            ground_truth = verdict,
                            source       = "ctu_win7ad",
                            raw_label    = raw_label,
                            service      = parts[7],
                            orig_port    = parts[3],
                            resp_port    = parts[5],
                            ts           = parts[0],
                            uid          = parts[1],
                            orig_h       = parts[2],
                            resp_h       = parts[4],
                            group_id     = member.name,
                        ))
                    except IndexError:
                        continue

                    if all(len(v) >= CAP for v in buckets.values()):
                        break
    except Exception as e:
        print(f"  [SKIP] Archive parse error: {e}")
        return []

    atk = len(buckets["ATTACK"])
    ben = len(buckets["FALSE POSITIVE"])
    print(f"  CTU-SME-11 Win7AD (OOD): {atk} attacks, {ben} benign sampled")
    return buckets["ATTACK"] + buckets["FALSE POSITIVE"]


def load_ctu_botnet3():
    """Load CTU-Malware Botnet-3 (Kelihos) as hard-floor OOD eval source.

    Kelihos is a P2P spam botnet captured in a malware sandbox. All flows
    in the Zeek conn.log originate from the infected VM and are ATTACK.
    The botnet is per-flow indistinguishable from normal web/email traffic —
    it scores MCC ~0.0 on all classifiers. Serves as the structural detection
    floor (no method can reliably detect it at the per-flow level).

    Local file: test_captures/CTU-Malware-Capture-Botnet-3_conn.log
    Standard 21-field Zeek conn.log (tab-separated, # header lines).
    """
    print(f"\n[CTU-Malware Botnet-3 OOD / Kelihos]")

    if not os.path.isfile(CTU_BOTNET3_LOG):
        print(f"  [SKIP] {CTU_BOTNET3_LOG} not found")
        return []

    samples = []
    with open(CTU_BOTNET3_LOG) as f:
        for line in f:
            if len(samples) >= CAP:
                break
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 21:
                continue
            samples.append(make_sample(
                proto        = parts[6],
                duration     = parts[8],
                orig_pkts    = parts[16],
                resp_pkts    = parts[18],
                orig_bytes   = parts[9],
                resp_bytes   = parts[10],
                conn_state   = parts[11],
                ground_truth = "ATTACK",
                source       = "ctu_botnet3",
                raw_label    = "Kelihos",
                service      = parts[7],
                orig_port    = parts[3],
                resp_port    = parts[5],
                ts           = parts[0],
                uid          = parts[1],
                orig_h       = parts[2],
                resp_h       = parts[4],
            ))

    print(f"  CTU-Malware Botnet-3 (OOD): {len(samples)} attacks, 0 benign")
    return samples
