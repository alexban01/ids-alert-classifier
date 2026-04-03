"""
bench_loaders.py — data loaders for benchmark_realworld.py.

Each loader returns a flat list of sample dicts (built with make_sample).
All loaders are self-contained: no ML imports, no GPU, safe to run in a
ProcessPoolExecutor worker.

Loaders:
    load_iot23(archive_path)      — IoT-23 conn.log.labeled from tar.gz
    load_ctu13(archive_path)      — CTU-13 binetflow CSV from tar.bz2
    load_uwf(dataset_dir)         — UWF-ZeekData24 Zeek conn.log CSVs
    load_ctu_normal(dataset_dir)  — CTU-Normal benign-only Zeek conn.log
    load_ctu_botnet90()           — CTU-Malware-Capture-Botnet-90 OOD probe (Conficker)
                                    (downloads from Stratosphere Lab if not cached)
"""

import os
import csv
import re
import tarfile
import glob
import urllib.request
from collections import defaultdict

import pandas as pd

from prompt_utils import build_prompt

# ── Constants ───────────────────────────────────────────────────────────────────
CAP = 3000   # max samples per (source, class)

# OOD regression test — CTU-Malware-Capture-Botnet-90 (Conficker) is intentionally
# removed from CTU_MALWARE_SCENARIOS so it is never included in training data.
# Conficker uses IRC C2 on non-standard port 2081 and LAN-scanning SYN probes —
# both are per-flow classifiable (unlike DarkVNC where individual TCP flows are
# indistinguishable from normal traffic). This makes it an informative OOD probe:
# if the model generalises botnet patterns from training, recall should be >0%.
CTU_BOTNET90_BASE_URL    = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-90"
CTU_BOTNET90_BINETFLOW  = "192.168.3.104-unvirus.binetflow"
CTU_BOTNET90_C2_IP      = "66.252.13.214"   # IRC C2 server s.unicat.org
# Port-based attack labeling fallback (used when binetflow has no Label column):
#   445  — SMB SYN scan (Conficker LAN spreading, 37k+ S0 flows)
#   2081 — IRC C2 channel on non-standard port
CTU_BOTNET90_ATTACK_PORTS = {"445", "2081"}
CTU_CAPTURE_DIR          = "test_captures"


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
    """Native Zeek conn.log.labeled from IoT-23 tar.gz.

    Last tab field bundles: tunnel_parents label detailed-label (space-separated).
    Detailed label examples: C&C, DDoS, Okiru, PartOfAHorizontalPortScan, etc.
    """
    if not os.path.isfile(archive_path):
        print(f"[SKIP] IoT-23 not found: {archive_path}")
        return []

    print(f"[IoT-23] Opening {archive_path} ...")
    buckets = defaultdict(list)

    with tarfile.open(archive_path, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.name.endswith("conn.log.labeled") and m.isfile()]
        print(f"  {len(members)} conn.log.labeled file(s)")

        for member in members:
            f = tf.extractfile(member)
            if f is None:
                continue
            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
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

                # Extract detailed label (3rd space-token in last field)
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
                        group_id   = member.name,
                    ))
                except IndexError:
                    continue

                # Stop early if both buckets are full
                if all(len(v) >= CAP for v in buckets.values()):
                    break

    atk = len(buckets["ATTACK"])
    ben = len(buckets["FALSE POSITIVE"])
    print(f"  IoT-23: {atk} attacks, {ben} benign")
    return buckets["ATTACK"] + buckets["FALSE POSITIVE"]


def load_ctu13(archive_path):
    """CTU-13 binetflow CSV from tar.bz2.

    Has TotPkts only (split 50/50 between orig/resp).
    Label: contains 'Botnet' → ATTACK, 'Normal' → FP, 'Background' → skip.
    """
    if not os.path.isfile(archive_path):
        print(f"[SKIP] CTU-13 not found: {archive_path}")
        return []

    print(f"[CTU-13] Opening {archive_path} ...")
    buckets = defaultdict(list)

    with tarfile.open(archive_path, "r:bz2") as tf:
        members = [m for m in tf.getmembers()
                   if m.name.endswith(".binetflow") and m.isfile()]
        print(f"  {len(members)} binetflow file(s)")

        for member in members:
            if all(len(v) >= CAP for v in [buckets["ATTACK"], buckets["FALSE POSITIVE"]]):
                break
            f = tf.extractfile(member)
            if f is None:
                continue
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
                    group_id   = member.name,
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


# ── CTU-Malware-Capture-Botnet-90 OOD loader ────────────────────────────────────

def _norm_key(proto, ip_a, port_a, ip_b, port_b):
    """Direction-agnostic 5-tuple key for binetflow ↔ conn.log matching."""
    pair_a = (str(ip_a).strip(), str(port_a).strip())
    pair_b = (str(ip_b).strip(), str(port_b).strip())
    lo, hi = (pair_a, pair_b) if pair_a <= pair_b else (pair_b, pair_a)
    return (str(proto).strip().lower(), lo[0], lo[1], hi[0], hi[1])


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


def load_ctu_botnet90():
    """Load CTU-Malware-Capture-Botnet-90 (Conficker) as permanent OOD eval source.

    Conficker uses IRC C2 on non-standard port 2081 (server: s.unicat.org /
    66.252.13.214) and performs LAN-wide SYN scanning. Both are per-flow
    classifiable — unlike DarkVNC where individual flows were indistinguishable
    from normal TCP. This scenario is intentionally NEVER included in training.

    Label strategy (in order):
      1. Binetflow Label column — looks for 'botnet'/'malware'/'attack' → ATTACK,
         'normal' → FALSE POSITIVE.
      2. C2 IP fallback — if binetflow has no Label column or yields 0 attacks,
         any flow to/from CTU_BOTNET90_C2_IP is labelled ATTACK, all others FP.
    """
    print(f"\n[CTU-Botnet-90 OOD / Conficker]")
    os.makedirs(CTU_CAPTURE_DIR, exist_ok=True)

    conn_local      = os.path.join(CTU_CAPTURE_DIR, "CTU-Botnet-90_conn.log")
    binetflow_local = os.path.join(CTU_CAPTURE_DIR, f"CTU-Botnet-90_{CTU_BOTNET90_BINETFLOW}")

    conn_path      = _bench_download(f"{CTU_BOTNET90_BASE_URL}/bro/conn.log", conn_local)
    binetflow_path = _bench_download(f"{CTU_BOTNET90_BASE_URL}/{CTU_BOTNET90_BINETFLOW}", binetflow_local)

    if conn_path is None:
        print("  [SKIP] conn.log download failed")
        return []

    # ── Try binetflow label matching ─────────────────────────────────────────────
    flow_labels     = {}
    use_ip_fallback = binetflow_path is None

    if not use_ip_fallback:
        try:
            with open(binetflow_path, newline="", errors="replace") as f:
                reader = csv.reader(f)
                header = None
                for row in reader:
                    if header is None:
                        header = [h.strip() for h in row]
                        idx    = {h: i for i, h in enumerate(header)}
                        needed = {"Proto", "SrcAddr", "Sport", "DstAddr", "Dport", "Label"}
                        if not needed.issubset(set(header)):
                            print(f"  binetflow has no Label column — using C2 IP fallback")
                            use_ip_fallback = True
                            break
                        continue
                    if len(row) <= max(idx["Label"], idx["Proto"], idx["SrcAddr"],
                                       idx["Sport"], idx["DstAddr"], idx["Dport"]):
                        continue
                    raw = row[idx["Label"]].strip().lower()
                    if "botnet" in raw or "malware" in raw or "attack" in raw:
                        label = "ATTACK"
                    elif "normal" in raw:
                        label = "FALSE POSITIVE"
                    else:
                        continue  # Background → skip
                    key = _norm_key(
                        row[idx["Proto"]],
                        row[idx["SrcAddr"]], row[idx["Sport"]],
                        row[idx["DstAddr"]], row[idx["Dport"]],
                    )
                    if key not in flow_labels or label == "ATTACK":
                        flow_labels[key] = label
        except Exception as e:
            print(f"  [WARN] binetflow parse error: {e} — using C2 IP fallback")
            use_ip_fallback = True

    if not use_ip_fallback:
        atk_n = sum(1 for v in flow_labels.values() if v == "ATTACK")
        ben_n = sum(1 for v in flow_labels.values() if v == "FALSE POSITIVE")
        print(f"  binetflow: {atk_n} ATTACK + {ben_n} FALSE POSITIVE labels")
        if atk_n == 0:
            print(f"  No ATTACK labels in binetflow — using C2 IP fallback")
            use_ip_fallback = True

    if use_ip_fallback:
        print(f"  Port fallback: resp_p in {CTU_BOTNET90_ATTACK_PORTS} → ATTACK "
              f"(445=SYN scan, 2081=IRC C2)")

    # ── Parse conn.log ───────────────────────────────────────────────────────────
    buckets = defaultdict(list)
    try:
        with open(conn_path, errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 19:
                    continue
                proto  = parts[6]
                orig_h = parts[2]; orig_p = parts[3]
                resp_h = parts[4]; resp_p = parts[5]

                if use_ip_fallback:
                    label = ("ATTACK" if resp_p in CTU_BOTNET90_ATTACK_PORTS
                             else "FALSE POSITIVE")
                else:
                    label = flow_labels.get(_norm_key(proto, orig_h, orig_p, resp_h, resp_p))
                    if label is None:
                        continue

                if len(buckets[label]) >= CAP:
                    continue

                buckets[label].append(make_sample(
                    proto        = proto,
                    duration     = parts[8],
                    orig_pkts    = parts[16] if len(parts) > 16 else "-",
                    resp_pkts    = parts[18] if len(parts) > 18 else "-",
                    orig_bytes   = parts[9],
                    resp_bytes   = parts[10],
                    conn_state   = parts[11],
                    ground_truth = label,
                    source       = "ctu_botnet90",
                    raw_label    = "Conficker" if label == "ATTACK" else "Benign",
                    service      = parts[7],
                    orig_port    = orig_p,
                    resp_port    = resp_p,
                    ts           = parts[0],
                    uid          = parts[1],
                    orig_h       = orig_h,
                    resp_h       = resp_h,
                    group_id     = "ctu_botnet90",
                ))

                if all(len(v) >= CAP for v in buckets.values()):
                    break
    except Exception as e:
        print(f"  [SKIP] conn.log parse error: {e}")
        return []

    atk = len(buckets["ATTACK"])
    ben = len(buckets["FALSE POSITIVE"])
    print(f"  Botnet-90 (OOD): {atk} attacks, {ben} benign sampled")
    return buckets["ATTACK"] + buckets["FALSE POSITIVE"]
