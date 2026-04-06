"""
analyze_gap.py — distribution gap analysis: training attacks vs Win7AD-1 OOD probe.

Usage:
    .venv/bin/python analyze_gap.py [--results results/benchmark_realworld_results.json]

Reads:
  - results/benchmark_realworld_cache.json   (bench samples with conn_state / resp_p)
  - results/benchmark_realworld_results.json (optional: per-sample OOD predictions + reasons)

Outputs three sections:
  1. conn_state distribution: ID training-proxy attacks vs Win7AD-1 attacks
  2. resp_port top-25: ID vs Win7AD-1
  3. Model prediction by conn_state for Win7AD-1 attacks (requires results JSON)
  4. Failure analysis table: top false negatives with REASON text
"""

import json
import os
import sys
from collections import Counter, defaultdict

CACHE_FILE   = "results/benchmark_realworld_cache.json"
RESULTS_JSON = "results/benchmark_realworld_results.json"

# Ports associated with Windows lateral movement
LATERAL_PORTS  = {88, 135, 139, 389, 445, 464, 593, 636, 3268, 3269, 3389, 49152}
TRICKBOT_PORTS = {4134, 22299}


def _port_int(v):
    try:
        return int(float(v or 0))
    except (ValueError, TypeError):
        return 0


def _port_flag(port):
    if port in LATERAL_PORTS:
        return "LATERAL"
    if port in TRICKBOT_PORTS:
        return "TRICKBOT-C2"
    return ""


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def analyze_conn_state(id_attacks, ood_attacks):
    section("conn_state Distribution: Training-proxy Attacks vs Win7AD-1 Attacks")

    id_cs  = Counter(s.get("conn_state", "-") or "-" for s in id_attacks)
    ood_cs = Counter(s.get("conn_state", "-") or "-" for s in ood_attacks)
    all_states = sorted(
        set(id_cs) | set(ood_cs),
        key=lambda x: -(id_cs[x] + ood_cs[x])
    )

    id_n  = sum(id_cs.values())  or 1
    ood_n = sum(ood_cs.values()) or 1

    print(f"\n  {'State':<12} {'Train n':>8} {'Train %':>8}   {'Win7AD n':>8} {'Win7AD %':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8}   {'-'*8} {'-'*8}")
    for state in all_states:
        marker = " <-- OOD dominant" if ood_cs[state] / ood_n > 0.20 and id_cs[state] / id_n < 0.05 else ""
        print(
            f"  {state:<12} {id_cs[state]:>8} {id_cs[state]/id_n:>8.1%}"
            f"   {ood_cs[state]:>8} {ood_cs[state]/ood_n:>8.1%}{marker}"
        )
    print(f"\n  Total:  {id_n} ID attacks  |  {ood_n} Win7AD-1 attacks")

    # Highlight the gap
    rej_id  = id_cs.get("REJ",  0)
    rstr_id = id_cs.get("RSTR", 0)
    rej_ood  = ood_cs.get("REJ",  0)
    rstr_ood = ood_cs.get("RSTR", 0)
    print(f"\n  KEY GAP: REJ+RSTR in training = {rej_id+rstr_id} ({(rej_id+rstr_id)/id_n:.1%})"
          f"  vs Win7AD-1 = {rej_ood+rstr_ood} ({(rej_ood+rstr_ood)/ood_n:.1%})")


def analyze_resp_port(id_attacks, ood_attacks):
    section("Destination Port Distribution: Training-proxy Attacks vs Win7AD-1 Attacks")

    id_ports  = Counter(_port_int(s.get("resp_p")) for s in id_attacks)
    ood_ports = Counter(_port_int(s.get("resp_p")) for s in ood_attacks)

    id_n  = sum(id_ports.values())  or 1
    ood_n = sum(ood_ports.values()) or 1

    # Union of top-20 from each side
    top_id  = {p for p, _ in id_ports.most_common(20)}
    top_ood = {p for p, _ in ood_ports.most_common(20)}
    all_ports = sorted(
        top_id | top_ood,
        key=lambda x: -(id_ports[x] + ood_ports[x])
    )

    print(f"\n  {'Port':<8} {'Train n':>8} {'Train %':>8}   {'Win7AD n':>8} {'Win7AD %':>8}  {'Flag'}")
    print(f"  {'-'*8} {'-'*8} {'-'*8}   {'-'*8} {'-'*8}  {'-'*12}")
    for port in all_ports[:30]:
        flag = _port_flag(port)
        marker = "  ***" if flag else ""
        print(
            f"  {port:<8} {id_ports[port]:>8} {id_ports[port]/id_n:>8.1%}"
            f"   {ood_ports[port]:>8} {ood_ports[port]/ood_n:>8.1%}"
            f"  {flag:<12}{marker}"
        )

    # Count lateral/Trickbot ports in each
    lat_id   = sum(id_ports[p] for p in LATERAL_PORTS)
    lat_ood  = sum(ood_ports[p] for p in LATERAL_PORTS)
    tbt_id   = sum(id_ports[p] for p in TRICKBOT_PORTS)
    tbt_ood  = sum(ood_ports[p] for p in TRICKBOT_PORTS)
    print(f"\n  Lateral ports (AD/RDP/SMB) in training: {lat_id} ({lat_id/id_n:.1%})"
          f"  vs Win7AD-1: {lat_ood} ({lat_ood/ood_n:.1%})")
    print(f"  Trickbot C2 ports in training:          {tbt_id} ({tbt_id/id_n:.1%})"
          f"  vs Win7AD-1: {tbt_ood} ({tbt_ood/ood_n:.1%})")


def analyze_predictions_by_conn_state(results_path):
    section("Model Predictions by conn_state — Win7AD-1 Attacks")

    if not os.path.exists(results_path):
        print(f"\n  [SKIP] {results_path} not found. Run benchmark first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    models = results.get("models", [])
    if not models:
        print("  [SKIP] No model results found.")
        return

    # Check if ood_samples is present (added in v10 benchmark changes)
    model = models[-1]
    ood_recs = model.get("ood_samples")
    if not ood_recs:
        print(f"\n  [SKIP] '{model['label']}' has no per-sample OOD records.")
        print("  Rerun benchmark_realworld.py to generate ood_samples data.")
        return

    # Filter to Win7AD-1 attacks only (source field added in v10; fall back for old results)
    atk_recs = [
        r for r in ood_recs
        if r["ground_truth"] == "ATTACK"
        and r.get("source", "ctu_win7ad") == "ctu_win7ad"
    ]
    if not atk_recs:
        print("  [SKIP] No Win7AD-1 attack records found.")
        return

    print(f"\n  Model: {model['label']}  |  {len(atk_recs)} Win7AD-1 attack samples\n")

    # Prediction rate by conn_state
    by_state = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in atk_recs:
        state = r.get("conn_state") or "-"
        by_state[state]["total"] += 1
        if r["prediction"] == "ATTACK":
            by_state[state]["correct"] += 1

    states_sorted = sorted(by_state, key=lambda x: -by_state[x]["total"])

    print(f"  {'conn_state':<12} {'n':>6} {'Detected':>9} {'Recall':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*9} {'-'*8}")
    for state in states_sorted:
        d = by_state[state]
        recall = d["correct"] / d["total"] if d["total"] else 0
        print(f"  {state:<12} {d['total']:>6} {d['correct']:>9} {recall:>8.1%}")


def failure_analysis_table(results_path, n=30):
    section(f"Failure Analysis — Win7AD-1 Attack False Negatives (top {n})")

    if not os.path.exists(results_path):
        print(f"\n  [SKIP] {results_path} not found.")
        return

    with open(results_path) as f:
        results = json.load(f)

    models = results.get("models", [])
    if not models:
        return

    model    = models[-1]
    ood_recs = model.get("ood_samples")
    if not ood_recs:
        print(f"\n  [SKIP] No per-sample OOD records in '{model['label']}'.")
        return

    false_negs = [
        r for r in ood_recs
        if r["ground_truth"] == "ATTACK"
        and r["prediction"] == "FALSE POSITIVE"
        and r.get("source", "ctu_win7ad") == "ctu_win7ad"
    ]
    if not false_negs:
        print("  No false negatives found (100% attack recall on Win7AD-1).")
        return

    # Sort by raw_label then conn_state for readable grouping
    false_negs.sort(key=lambda r: (r.get("raw_label", ""), r.get("conn_state", "")))

    print(f"\n  Model: {model['label']}  |  {len(false_negs)} false negatives\n")
    print(f"  {'raw_label':<46} {'State':<8} {'Port':<7}  REASON")
    print(f"  {'-'*46} {'-'*8} {'-'*7}  {'-'*80}")
    for r in false_negs[:n]:
        state  = r.get("conn_state") or "-"
        port   = str(_port_int(r.get("resp_p")))
        label  = (r.get("raw_label") or "-")[:45]
        reason = (r.get("reason") or "")
        print(f"  {label:<46} {state:<8} {port:<7}  {reason}")

    if len(false_negs) > n:
        print(f"\n  ... {len(false_negs) - n} more false negatives not shown")

    # Summary by raw_label
    print(f"\n  False negatives by attack type:")
    by_label = Counter(r.get("raw_label", "-") for r in false_negs)
    for label, count in by_label.most_common():
        print(f"    {label:<48} {count}")


if __name__ == "__main__":
    results_path = RESULTS_JSON
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--results" and i + 1 < len(args):
            results_path = args[i + 1]

    if not os.path.exists(CACHE_FILE):
        print(f"[ERROR] Bench cache not found: {CACHE_FILE}")
        print("  Run: .venv/bin/python benchmark_realworld.py --regen")
        sys.exit(1)

    print(f"Loading {CACHE_FILE} ...")
    with open(CACHE_FILE) as f:
        samples = json.load(f)

    # Training proxy: in-distribution attack samples from the 4 training sources
    # (CTU-Malware is not in the bench cache, but IoT-23/CTU-13/UWF represent the
    # same conn_state/port distribution — all dominated by inbound scans and botnet C2)
    id_attacks  = [s for s in samples
                   if s["source"] in ("iot23", "ctu13", "uwf")
                   and s["ground_truth"] == "ATTACK"]
    ood_attacks = [s for s in samples
                   if s["source"] == "ctu_win7ad"
                   and s["ground_truth"] == "ATTACK"]

    print(f"  ID attack samples (train proxy): {len(id_attacks)}")
    print(f"  Win7AD-1 attack samples (OOD):   {len(ood_attacks)}")

    if not ood_attacks:
        print("[ERROR] No Win7AD-1 attack samples in cache. Run benchmark with --regen --ood first.")
        sys.exit(1)

    analyze_conn_state(id_attacks, ood_attacks)
    analyze_resp_port(id_attacks, ood_attacks)
    analyze_predictions_by_conn_state(results_path)
    failure_analysis_table(results_path)

    print(f"\n{'='*72}")
    print("  Done. Use these tables in the thesis gap analysis section.")
    print(f"{'='*72}\n")
