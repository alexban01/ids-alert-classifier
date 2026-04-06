"""
baseline_ml.py — Random Forest + Logistic Regression baseline for IDS classifier.

Trains on in-distribution sources from the benchmark cache, evaluates on all sources
including OOD probes. Produces the same per-source metrics as benchmark_realworld.py
for direct table comparison.

Usage:
    .venv/bin/python baseline_ml.py [--cache FILE] [--no-cv]

    --cache FILE    Path to benchmark cache JSON (default: results/benchmark_realworld_cache.json)
    --no-cv         Skip cross-validation, just train on 80% of ID sources

Features (14): 7 numeric + 7 categorical ordinal
  Numeric:     duration, orig_pkts, resp_pkts, orig_bytes, resp_bytes,
               bytes_per_sec, orig_bytes_per_pkt
  Categorical: proto, conn_state, service, resp_port (int), resp_port_tier,
               resp_port_is_known, src_port_tier
"""

import json
import os
import sys
import random
import warnings
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=FutureWarning)

CACHE_FILE = "results/benchmark_realworld_cache.json"
RANDOM_SEED = 42

# Sources that are OOD probes — held out from training entirely
OOD_SOURCES = {"ctu_win7ad", "ctu_sme11", "ctu_botnet3"}

SOURCE_NAMES = {
    "iot23":        "IoT-23          (Zeek conn.log)",
    "ctu13":        "CTU-13          (binetflow)",
    "uwf":          "UWF-ZeekData24  (Zeek conn.log)",
    "ctu_normal":   "CTU-Normal      (Zeek conn.log)",
    "ctu_botnet90": "CTU-Botnet90    (binetflow)",
    "ctu_mixed6":   "CTU-Mixed6      (binetflow)",
    "ctu_win7ad":   "CTU-SME-11 [OOD] (Windows7AD-1)",
    "ctu_sme11":    "CTU-SME-11 [OOD] (Amazon Echo)",
    "ctu_botnet3":  "CTU-Botnet3 [OOD] (Kelihos floor)",
}

# Known ports for resp_port_is_known feature
KNOWN_PORTS = {
    20, 21, 22, 23, 25, 53, 67, 68, 80, 88, 110, 123, 135, 137, 138,
    139, 143, 161, 389, 443, 445, 464, 465, 587, 593, 636, 993, 995,
    1433, 1521, 3268, 3269, 3306, 3389, 5432, 5900, 6379, 6667, 8080,
    8443, 27017,
}

PROTO_MAP = {"tcp": 0, "udp": 1, "icmp": 2}

CONN_STATE_MAP = {
    # Zeek native
    "SF": 0, "S0": 1, "RSTO": 2, "RSTR": 3, "REJ": 4, "OTH": 5,
    "S1": 6, "S2": 7, "S3": 8, "SHR": 9, "RSTOS0": 10, "RSTRH": 11,
    # Argus/binetflow mappings used in CTU-13 loader
    "INT": 6, "CON": 0,
}

SERVICE_MAP = {
    "http": 0, "dns": 1, "ssl": 2, "irc": 3, "ftp": 4, "ssh": 5,
    "smtp": 6, "pop3": 7, "imap": 8, "ftp-data": 9,
}


def _safe_float(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _safe_port(v):
    try:
        return int(float(v or 0))
    except (ValueError, TypeError):
        return 0


def extract_features(sample):
    """Convert a cache sample dict to a 14-element float feature vector."""
    dur   = max(_safe_float(sample.get("duration")), 0.0)
    op    = max(_safe_float(sample.get("orig_pkts")), 0.0)
    rp    = max(_safe_float(sample.get("resp_pkts")), 0.0)
    ob    = max(_safe_float(sample.get("orig_bytes")), 0.0)
    rb    = max(_safe_float(sample.get("resp_bytes")), 0.0)

    bps        = (ob + rb) / dur if dur > 0 else 0.0
    ob_per_pkt = ob / op if op > 0 else 0.0

    proto_ord = PROTO_MAP.get(str(sample.get("proto", "")).lower(), 3)

    cs = str(sample.get("conn_state", "") or "")
    conn_ord = CONN_STATE_MAP.get(cs, 12)  # 12 = unknown

    svc = str(sample.get("service", "") or "").lower()
    svc = svc if svc not in ("-", "", "none", "nan") else "-"
    svc_ord = SERVICE_MAP.get(svc, 10)     # 10 = other/unknown

    resp_port    = _safe_port(sample.get("resp_p"))
    orig_port    = _safe_port(sample.get("orig_p"))

    if resp_port < 1024:
        resp_tier = 0
    elif resp_port < 49152:
        resp_tier = 1
    else:
        resp_tier = 2

    if orig_port < 1024:
        src_tier = 0
    elif orig_port < 49152:
        src_tier = 1
    else:
        src_tier = 2

    resp_known = 1 if resp_port in KNOWN_PORTS else 0

    return [
        dur, op, rp, ob, rb, bps, ob_per_pkt,
        proto_ord, conn_ord, svc_ord,
        float(resp_port), float(resp_tier), float(resp_known), float(src_tier),
    ]


def load_data(cache_path):
    with open(cache_path) as f:
        samples = json.load(f)

    id_samples  = [s for s in samples if s["source"] not in OOD_SOURCES]
    ood_samples = {src: [s for s in samples if s["source"] == src]
                   for src in OOD_SOURCES}

    X_id = np.array([extract_features(s) for s in id_samples], dtype=np.float32)
    y_id = np.array([1 if s["ground_truth"] == "ATTACK" else 0 for s in id_samples])

    return X_id, y_id, id_samples, ood_samples


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def per_source_report(model_label, preds, samples, sources_to_show):
    print(f"\n  {'Source':<36} {'Atk':>5} {'Recall':>7}   {'Ben':>5} {'Recall':>7}   {'Acc':>6}   {'MCC':>7}")
    print(f"  {'-'*36} {'-'*5} {'-'*7}   {'-'*5} {'-'*7}   {'-'*6}   {'-'*7}")

    overall_t, overall_p = [], []
    for src in sources_to_show:
        idx   = [i for i, s in enumerate(samples) if s["source"] == src]
        if not idx:
            continue
        t_sub = [("ATTACK" if samples[i]["ground_truth"] == "ATTACK" else "FALSE POSITIVE") for i in idx]
        p_sub = [("ATTACK" if preds[i] == 1 else "FALSE POSITIVE") for i in idx]
        a_idx = [i for i, t in enumerate(t_sub) if t == "ATTACK"]
        b_idx = [i for i, t in enumerate(t_sub) if t == "FALSE POSITIVE"]
        a_rec = sum(p_sub[i] == "ATTACK"         for i in a_idx) / max(len(a_idx), 1)
        b_rec = sum(p_sub[i] == "FALSE POSITIVE" for i in b_idx) / max(len(b_idx), 1)
        acc   = sum(t == p for t, p in zip(t_sub, p_sub)) / len(t_sub)
        mcc   = matthews_corrcoef(
            [1 if t == "ATTACK" else 0 for t in t_sub],
            [1 if p == "ATTACK" else 0 for p in p_sub],
        )
        name  = SOURCE_NAMES.get(src, src)
        print(f"  {name:<36} {len(a_idx):>5} {a_rec:>7.1%}   {len(b_idx):>5} {b_rec:>7.1%}   {acc:>6.1%}   {mcc:>+7.3f}")
        overall_t.extend([1 if t == "ATTACK" else 0 for t in t_sub])
        overall_p.extend([1 if p == "ATTACK" else 0 for p in p_sub])

    if overall_t:
        total_mcc = matthews_corrcoef(overall_t, overall_p)
        acc = sum(t == p for t, p in zip(overall_t, overall_p)) / len(overall_t)
        atk_r = sum(1 for t, p in zip(overall_t, overall_p) if t == 1 and p == 1) / max(sum(overall_t), 1)
        ben_r = sum(1 for t, p in zip(overall_t, overall_p) if t == 0 and p == 0) / max(overall_t.count(0), 1)
        n_atk = sum(overall_t)
        n_ben = len(overall_t) - n_atk
        print(f"\n  {'OVERALL':<36} {n_atk:>5} {atk_r:>7.1%}   {n_ben:>5} {ben_r:>7.1%}   {acc:>6.1%}   {total_mcc:>+7.3f}")


def run_baseline(X_train, y_train, X_test_id, id_samples, ood_samples, do_cv):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    lr = LogisticRegression(
        C=0.1, class_weight="balanced",
        max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED,
    )

    print(f"\nTraining RF  on {len(X_train)} samples...")
    rf.fit(X_train, y_train)
    print(f"Training LR  on {len(X_train)} samples...")
    lr.fit(X_train_s, y_train)

    # ── In-distribution test ──────────────────────────────────────────────────
    section("In-Distribution Test (80/20 split within ID sources)")
    test_idx  = list(range(len(X_test_id)))
    X_test_s  = scaler.transform(X_test_id)
    rf_preds  = rf.predict(X_test_id)
    lr_preds  = lr.predict(X_test_s)
    id_sources_order = sorted(set(s["source"] for s in id_samples))

    for label, preds in [("Random Forest", rf_preds), ("Logistic Regression", lr_preds)]:
        t_labels = [1 if s["ground_truth"] == "ATTACK" else 0 for s in id_samples]
        mcc = matthews_corrcoef(t_labels, preds)
        acc = sum(t == p for t, p in zip(t_labels, preds)) / len(t_labels)
        atk_r = sum(1 for t, p in zip(t_labels, preds) if t == 1 and p == 1) / max(sum(t_labels), 1)
        ben_r = sum(1 for t, p in zip(t_labels, preds) if t == 0 and p == 0) / max(t_labels.count(0), 1)
        print(f"\n  {label}: MCC={mcc:+.4f}  Accuracy={acc:.1%}  Atk Recall={atk_r:.1%}  FP Recall={ben_r:.1%}")
        per_source_report(label, preds, id_samples, id_sources_order)

    # ── OOD evaluation ────────────────────────────────────────────────────────
    section("OOD Evaluation — Primary (Windows7AD-1)")
    _ood_eval(rf, lr, scaler, ood_samples, "ctu_win7ad")

    section("OOD Evaluation — Easy (Amazon Echo)")
    _ood_eval(rf, lr, scaler, ood_samples, "ctu_sme11")

    section("OOD Evaluation — Floor (Kelihos Botnet-3)")
    _ood_eval(rf, lr, scaler, ood_samples, "ctu_botnet3")

    # ── Feature importance (RF) ───────────────────────────────────────────────
    section("RF Feature Importance")
    feat_names = [
        "duration", "orig_pkts", "resp_pkts", "orig_bytes", "resp_bytes",
        "bytes_per_sec", "orig_bytes_per_pkt",
        "proto", "conn_state", "service",
        "resp_port", "resp_port_tier", "resp_port_known", "src_port_tier",
    ]
    importances = sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1])
    print()
    for name, imp in importances:
        bar = "█" * int(imp * 80)
        print(f"  {name:<20} {imp:.4f}  {bar}")


def _ood_eval(rf, lr, scaler, ood_samples, src_key):
    samples = ood_samples.get(src_key, [])
    if not samples:
        print(f"\n  [SKIP] No samples for {src_key} in cache.")
        return

    X = np.array([extract_features(s) for s in samples], dtype=np.float32)
    y = np.array([1 if s["ground_truth"] == "ATTACK" else 0 for s in samples])
    X_s = scaler.transform(X)

    for label, preds in [("Random Forest", rf.predict(X)), ("Logistic Regression", lr.predict(X_s))]:
        mcc    = matthews_corrcoef(y, preds)
        n_atk  = sum(y)
        n_ben  = len(y) - n_atk
        atk_r  = sum(1 for t, p in zip(y, preds) if t == 1 and p == 1) / max(n_atk, 1)
        ben_r  = sum(1 for t, p in zip(y, preds) if t == 0 and p == 0) / max(n_ben, 1)
        acc    = sum(t == p for t, p in zip(y, preds)) / len(y)
        name   = SOURCE_NAMES.get(src_key, src_key)
        print(f"\n  {label} on {name}")
        print(f"    MCC={mcc:+.4f}  Accuracy={acc:.1%}  Atk Recall={atk_r:.1%}  FP Recall={ben_r:.1%}  n={len(y)}")

        # Per raw_label breakdown for win7ad
        if src_key == "ctu_win7ad":
            by_label = {}
            for s, p in zip(samples, preds):
                if s["ground_truth"] != "ATTACK":
                    continue
                rl = s.get("raw_label", "-")
                if rl not in by_label:
                    by_label[rl] = {"total": 0, "correct": 0}
                by_label[rl]["total"] += 1
                if p == 1:
                    by_label[rl]["correct"] += 1
            for rl, d in sorted(by_label.items()):
                recall = d["correct"] / d["total"] if d["total"] else 0
                print(f"    {rl[:60]:<60}  {d['correct']:>3}/{d['total']:<4} ({recall:.0%})")


if __name__ == "__main__":
    cache_path = CACHE_FILE
    do_cv      = True
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--cache"  and i + 1 < len(args): cache_path = args[i + 1]
        if arg == "--no-cv": do_cv = False

    if not os.path.exists(cache_path):
        print(f"[ERROR] Cache not found: {cache_path}")
        print("  Run: .venv/bin/python benchmark_realworld.py --regen")
        sys.exit(1)

    print(f"Loading {cache_path} ...")
    X_id, y_id, id_samples, ood_samples = load_data(cache_path)

    id_src_counts = Counter(s["source"] for s in id_samples)
    ood_src_counts = {k: len(v) for k, v in ood_samples.items()}
    print(f"  ID sources: {dict(id_src_counts)}")
    print(f"  OOD sources: {ood_src_counts}")
    print(f"  ID total: {len(X_id)} samples  ({sum(y_id)} attacks / {len(y_id)-sum(y_id)} benign)")

    # 80/20 stratified split within ID sources
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(skf.split(X_id, y_id))  # use first fold (80/20)

    X_train    = X_id[train_idx]
    y_train    = y_id[train_idx]
    X_test_id  = X_id[test_idx]
    id_test_samples = [id_samples[i] for i in test_idx]

    print(f"  Train: {len(X_train)}  Test (ID): {len(X_test_id)}")

    run_baseline(X_train, y_train, X_test_id, id_test_samples, ood_samples, do_cv)
