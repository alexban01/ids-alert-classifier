"""
preprocess_zeek.py — Build training JSONL from Zeek-native / Zeek-compatible datasets.

Outputs:
  zeek_dataset.jsonl       — training samples (~90% per source/class bucket)
  zeek_dataset_eval.jsonl  — held-out eval samples (~10%, source-stratified)

Module layout
─────────────
  preprocess_config.py     — caps, ratio targets, masking probs, reason pools
  preprocess_sample.py     — score_hard_benign(), make_sample(), pick_reason()
  zeek_log_utils.py        — Zeek TSV parser + CTU-Malware download helpers
  loader_iot23.py          — IoT-23 conn.log.labeled (tar.gz)
  loader_ctu13.py          — CTU-13 binetflow (tar.bz2)
  loader_unsw.py           — UNSW-NB15 parquet / CSV
  loader_cicids.py         — CICIDS2017 CICFlowMeter CSVs (disabled in v7+)
  loader_uwf.py            — UWF-ZeekData24 Spark CSV
  loader_ctu_normal.py     — CTU-Normal benign Zeek conn.log
  loader_ctu_malware.py    — CTU-Malware-Capture multi-log enriched samples
"""

import json
import random
from collections import Counter, defaultdict

from preprocess_config import (
    DATASETS,
    EVAL_FILE,
    EVAL_FRAC,
    FINAL_ATTACK,
    FINAL_BENIGN,
    HARD_BENIGN_MIN_SCORE,
    HARD_BENIGN_TARGET_FRAC,
    RANDOM_SEED,
    TRAIN_FILE,
)
from loader_iot23         import load_iot23
from loader_ctu13         import load_ctu13
from loader_unsw          import load_unsw
from loader_cicids        import load_cicids          # noqa: F401 — kept for optional re-enable
from loader_uwf           import load_uwf
from loader_ctu_normal    import load_ctu_normal
from loader_ctu_malware   import load_ctu_malware_captures


def _write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps({"messages": s["messages"]}) + "\n")


def _ctx_pct(pool, key):
    """Fraction (%) of samples in pool whose user prompt contains key."""
    has_ctx = sum(1 for s in pool
                  if any(key in msg["content"]
                         for msg in s["messages"] if msg["role"] == "user"))
    return 100 * has_ctx / max(len(pool), 1)


if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    # ── Load all sources ──────────────────────────────────────────────────────
    all_samples = []
    all_samples += load_iot23(DATASETS["iot23"])
    all_samples += load_ctu13(DATASETS["ctu13"])
    all_samples += load_unsw(DATASETS["unsw"])
    # CICIDS2017 disabled in v7: CICFlowMeter produces proto="unknown" and
    # conn_state="-" for every flow — both most-discriminative Zeek features
    # are always absent.  Also has ~10-15% label errors.
    # all_samples += load_cicids(DATASETS["cicids"])
    all_samples += load_uwf(DATASETS["uwf"])
    all_samples += load_ctu_normal(DATASETS["ctu_normal"])
    # v9.0: CTU-Malware-Capture — multi-log enriched training samples.
    # Botnet-3 (Kelihos) held out as OOD test — not included here.
    all_samples += load_ctu_malware_captures()

    # ── Source-stratified train/eval split ────────────────────────────────────
    # Hold out EVAL_FRAC from each (source, verdict) bucket so eval distribution
    # mirrors the full source variety rather than a random 10% slice.
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
    print(f"Hard benigns  : {sum(1 for s in benign if s.get('is_hard_benign'))} "
          f"(score >= {HARD_BENIGN_MIN_SCORE})")

    # SF-state attack oversampling: completed/established attacks get 2× weight
    # to address near-zero recall on SF-state attacks (Credential Access, HTTP
    # C2, exfil).  S0/SYN attacks keep 1× weight.  k is capped at pool size
    # so small TRAINING_FACTOR runs don't duplicate entries.
    sf_attacks    = [s for s in attacks if s.get("conn_state", "-") in ("SF", "S1", "OTH")]
    other_attacks = [s for s in attacks if s.get("conn_state", "-") not in ("SF", "S1", "OTH")]
    weights       = [2.0] * len(sf_attacks) + [1.0] * len(other_attacks)
    k_attacks     = min(FINAL_ATTACK, len(attacks))
    print(f"  SF/S1/OTH attacks (2× weight): {len(sf_attacks):,} | other: {len(other_attacks):,}")
    attacks = random.choices(sf_attacks + other_attacks, weights=weights, k=k_attacks)

    # 2:1 ratio: benign target is 2× actual attacks taken, capped at pool size.
    # Reserve HARD_BENIGN_TARGET_FRAC of the benign budget for hard negatives
    # (flows that look attack-like by state/port/context/behavior).
    k_benign   = min(FINAL_BENIGN, 2 * k_attacks, len(benign))
    hard_benign  = [s for s in benign if s.get("is_hard_benign")]
    other_benign = [s for s in benign if not s.get("is_hard_benign")]

    if len(benign) <= k_benign:
        random.shuffle(benign)
    else:
        random.shuffle(hard_benign)
        hard_benign.sort(key=lambda s: s.get("hard_benign_score", 0), reverse=True)
        hard_keep      = min(len(hard_benign), int(k_benign * HARD_BENIGN_TARGET_FRAC))
        selected_hard  = hard_benign[:hard_keep]
        remaining      = k_benign - len(selected_hard)
        random.shuffle(other_benign)
        benign = selected_hard + other_benign[:remaining]
        random.shuffle(benign)

    final_train = attacks + benign
    random.shuffle(final_train)
    random.shuffle(eval_pool)

    # ── Write outputs ─────────────────────────────────────────────────────────
    _write_jsonl(TRAIN_FILE, final_train)
    _write_jsonl(EVAL_FILE,  eval_pool)

    # ── Context coverage diagnostics ──────────────────────────────────────────
    atk_pool = [s for s in final_train if s["verdict"] == "ATTACK"]
    ben_pool = [s for s in final_train if s["verdict"] == "FALSE POSITIVE"]
    for section in ("[HTTP]", "[DNS]", "[SSL]", "[BEHAVIOR]"):
        ap   = _ctx_pct(atk_pool, section)
        bp   = _ctx_pct(ben_pool, section)
        flag = " ⚠ imbalanced" if ap > 0 and bp == 0 else ""
        print(f"   Context {section}: atk {ap:.1f}% / ben {bp:.1f}%{flag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    train_hard_benign = [s for s in final_train
                         if s["verdict"] == "FALSE POSITIVE" and s.get("is_hard_benign")]
    print(f"\n✅ {len(final_train)} train samples → {TRAIN_FILE}")
    print(f"   Attacks: {len(attacks):>7,}  |  Benign: {len(benign):>7,}  "
          f"(ratio 1:{len(benign)/max(len(attacks),1):.1f})")
    print(f"   Hard benign kept: {len(train_hard_benign):>7,}  "
          f"({100*len(train_hard_benign)/max(len(benign),1):.1f}% of benign train)")
    print(f"✅ {len(eval_pool)} eval samples  → {EVAL_FILE}")

    print(f"\n   Train source breakdown:")
    sources = Counter(s["source"] for s in final_train)
    for src, n in sorted(sources.items()):
        a = sum(1 for s in final_train if s["source"] == src and s["verdict"] == "ATTACK")
        b = sum(1 for s in final_train if s["source"] == src and s["verdict"] == "FALSE POSITIVE")
        print(f"   {src:12s}: {n:>7,}  (atk {a:>6,} / ben {b:>6,})")

    print(f"\n   Hard benign source breakdown:")
    hb_sources = Counter(s["source"] for s in train_hard_benign)
    for src, n in sorted(hb_sources.items()):
        avg_score = (sum(s.get("hard_benign_score", 0) for s in train_hard_benign
                         if s["source"] == src) / max(n, 1))
        print(f"   {src:12s}: {n:>7,}  (avg score {avg_score:.1f})")
