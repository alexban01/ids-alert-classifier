"""
preprocess_zeek.py — Build training JSONL from Zeek-native / Zeek-compatible datasets.

Outputs:
  zeek_dataset.jsonl       — training samples (~90% per source/class bucket)
  zeek_dataset_eval.jsonl  — held-out eval samples (~10%, source-stratified)

Module layout
─────────────
  preprocess_config.py            — caps, ratio targets, masking probs, reason pools
  preprocess_sample.py            — score_hard_benign(), make_sample(), pick_reason()
  zeek_log_utils.py               — Zeek TSV parser + CTU-Malware download helpers
  loaders/loader_iot23.py         — IoT-23 conn.log.labeled (tar.gz)
  loaders/loader_ctu13.py         — CTU-13 binetflow (tar.bz2)
  loaders/loader_unsw.py          — UNSW-NB15 parquet / CSV
  loaders/loader_cicids.py        — CICIDS2017 CICFlowMeter CSVs (disabled in v7+)
  loaders/loader_uwf.py           — UWF-ZeekData24 Spark CSV
  loaders/loader_ctu_normal.py    — CTU-Normal benign Zeek conn.log
  loaders/loader_ctu_malware.py   — CTU-Malware-Capture multi-log enriched samples
"""

import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from ids.preprocess_config import (
    CTU_MALWARE_ATTACK_BUDGET,
    CTU_MALWARE_SCENARIOS,
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
from ids.loaders.loader_iot23         import load_iot23_file
from ids.loaders.loader_ctu13         import load_ctu13_file
from ids.loaders.loader_unsw          import load_unsw
from ids.loaders.loader_cicids        import load_cicids          # noqa: F401 — kept for optional re-enable
from ids.loaders.loader_uwf           import load_uwf
from ids.loaders.loader_ctu_normal    import load_ctu_normal
from ids.loaders.loader_ctu_malware   import load_ctu_malware_scenario

def _run_loader_job(job_name, dataset_path):
    """Worker wrapper for ProcessPoolExecutor loader jobs.

    Redirects stdout to /dev/null so per-loader print output doesn't
    interleave with the tqdm bar in the main process.
    """
    import sys
    sys.stdout = open(os.devnull, "w")
    random.seed(RANDOM_SEED)
    if job_name == "iot23_single":
        return job_name, load_iot23_file(dataset_path)
    if job_name == "ctu13_single":
        return job_name, load_ctu13_file(dataset_path)
    if job_name == "unsw":
        return job_name, load_unsw(dataset_path)
    if job_name == "uwf":
        return job_name, load_uwf(dataset_path)
    if job_name == "ctu_normal":
        return job_name, load_ctu_normal(dataset_path)
    if job_name == "ctu_malware_single":
        scenario_id, family, base_url = dataset_path
        return job_name, load_ctu_malware_scenario(scenario_id, family, base_url)
    raise ValueError(f"Unknown loader job: {job_name}")


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ctu-only", action="store_true",
        help="Skip IoT-23 and UNSW-NB15 (slow/large); keep CTU-13, CTU-Normal, CTU-Malware, UWF.",
    )
    parser.add_argument(
        "--no-uwf", action="store_true",
        help="Also skip UWF-ZeekData24 (combine with --ctu-only for CTU-only benign sources).",
    )
    parser.add_argument(
        "--no-reason", action="store_true",
        help="Drop the REASON line from training targets (verdict-only ablation). "
             "Samples become 'VERDICT: <X>' with a verdict-only system prompt. "
             "Benchmark/deploy the resulting model with SYSTEM_PROMPT_VERDICT_ONLY.",
    )
    args = parser.parse_args()

    # Set before the ProcessPoolExecutor is created so loader workers (separate
    # processes) inherit it — make_sample reads IDS_NO_REASON from the env.
    if args.no_reason:
        os.environ["IDS_NO_REASON"] = "1"
        print("[INFO] --no-reason: training targets are verdict-only (REASON line dropped)")

    random.seed(RANDOM_SEED)

    # ── Load all sources ──────────────────────────────────────────────────────
    # Each CTU-13 binetflow file and each CTU-Malware scenario is its own
    # parallel job — independent parse with no shared state.
    _iot23_files = sorted(
        os.path.join(root, fname)
        for root, _, files in os.walk(DATASETS["iot23"])
        for fname in files
        if fname == "conn.log.labeled"
    )
    if not _iot23_files:
        print(f"[WARN] No conn.log.labeled files found under {DATASETS['iot23']} — skipping")

    _ctu13_files = sorted(
        os.path.join(root, fname)
        for root, _, files in os.walk(DATASETS["ctu13"])
        for fname in files
        if fname.endswith(".binetflow")
    )
    if not _ctu13_files:
        print(f"[WARN] No CTU-13 binetflow files found in {DATASETS['ctu13']} — skipping")

    loader_jobs = [
        ("unsw",       DATASETS["unsw"]),
        ("uwf",        DATASETS["uwf"]),
        ("ctu_normal", DATASETS["ctu_normal"]),
    ] + [
        ("iot23_single", fp) for fp in _iot23_files
    ] + [
        ("ctu13_single", fp) for fp in _ctu13_files
    ] + [
        ("ctu_malware_single", (sid, fam, url))
        for sid, fam, url in CTU_MALWARE_SCENARIOS
    ]
    if args.ctu_only:
        loader_jobs = [(n, p) for n, p in loader_jobs if n not in ("iot23_single", "unsw")]
        print("[INFO] --ctu-only: skipping IoT-23 and UNSW-NB15")
    if args.no_uwf:
        loader_jobs = [(n, p) for n, p in loader_jobs if n != "uwf"]
        print("[INFO] --no-uwf: skipping UWF-ZeekData24")
    max_workers = min(len(loader_jobs), os.cpu_count() or 1)
    all_samples = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_loader_job, job_name, dataset_path): job_name
            for job_name, dataset_path in loader_jobs
        }
        with tqdm(total=len(loader_jobs), desc="Loading", unit="jobs") as pbar:
            for future in as_completed(futures):
                job_name = futures[future]
                try:
                    _, job_samples = future.result()
                except Exception as e:
                    tqdm.write(f"[ERROR] Loader '{job_name}' failed: {e}")
                    job_samples = []
                all_samples.extend(job_samples)
                tqdm.write(f"[DONE] {job_name}: {len(job_samples):,} samples"
                           f"  (total so far: {len(all_samples):,})")
                pbar.update(1)

    # CICIDS2017 disabled in v7: CICFlowMeter produces proto="unknown" and
    # conn_state="-" for every flow — both most-discriminative Zeek features
    # are always absent.  Also has ~10-15% label errors.
    # all_samples += load_cicids(DATASETS["cicids"])

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

    # Source-stratified attack sampling: guarantee CTU-Malware representation.
    # Without this, CTU-Malware falls to ~19% after the weighted draw because
    # IoT-23/CTU-13 pools are larger.  Target: 40% (48k / 120k) at full scale.
    ctu_m  = [s for s in attacks if s["source"] == "ctu_malware"]
    others = [s for s in attacks if s["source"] != "ctu_malware"]

    n_m = min(CTU_MALWARE_ATTACK_BUDGET, len(ctu_m))
    n_o = min(FINAL_ATTACK - n_m, len(others))
    print(f"  CTU-Malware pool: {len(ctu_m):,} → taking {n_m:,}"
          f"  (budget={CTU_MALWARE_ATTACK_BUDGET:,})")

    # CTU-Malware: simple shuffle + slice (pool already has diverse states)
    random.shuffle(ctu_m)
    ctu_m_selected = ctu_m[:n_m]

    # Other attacks: SF oversampling (2× weight for SF/S1/OTH) to address
    # near-zero recall on completed-connection attacks (Credential Access, C2).
    sf_others    = [s for s in others if s.get("conn_state", "-") in ("SF", "S1", "OTH")]
    other_others = [s for s in others if s.get("conn_state", "-") not in ("SF", "S1", "OTH")]
    weights      = [2.0] * len(sf_others) + [1.0] * len(other_others)
    print(f"  Other attacks — SF/S1/OTH (2× weight): {len(sf_others):,} | other: {len(other_others):,}"
          f" → taking {n_o:,}")
    other_selected = random.choices(sf_others + other_others, weights=weights, k=n_o) if others else []

    # ── Per-source hard cap on the weighted draw ───────────────────────────────
    # random.choices (with replacement) + 2× SF weight can push SF-heavy sources
    # (e.g. UWF Credential Access) above the composition threshold even when
    # loader caps are respected.  Trim to 25% of FINAL_ATTACK per source, then
    # refill the shortfall from sources still under cap.
    _src_draw_cap = int(FINAL_ATTACK * 0.25)
    _src_drawn    = Counter(s["source"] for s in other_selected)
    if any(c > _src_draw_cap for c in _src_drawn.values()):
        random.shuffle(other_selected)
        _seen, _kept = Counter(), []
        for s in other_selected:
            if _seen[s["source"]] < _src_draw_cap:
                _kept.append(s)
                _seen[s["source"]] += 1
        shortfall = n_o - len(_kept)
        if shortfall > 0:
            _refill_pool = [s for s in (sf_others + other_others)
                            if _seen[s["source"]] < _src_draw_cap]
            if _refill_pool:
                _rw = [2.0 if s.get("conn_state", "-") in ("SF", "S1", "OTH") else 1.0
                       for s in _refill_pool]
                _kept.extend(random.choices(_refill_pool, weights=_rw, k=shortfall))
        other_selected = _kept

    attacks = ctu_m_selected + other_selected
    random.shuffle(attacks)

    # 2:1 ratio: benign target is 2× actual attacks taken, capped at pool size.
    # Reserve HARD_BENIGN_TARGET_FRAC of the benign budget for hard negatives
    # (flows that look attack-like by state/port/context/behavior).
    k_benign   = min(FINAL_BENIGN, 2 * len(attacks), len(benign))
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
        selected_other = other_benign[:remaining]
        # Non-hard pool exhausted — fill the rest from remaining hard benigns (score-ranked)
        if len(selected_other) < remaining:
            extra_hard     = hard_benign[hard_keep:]
            fill           = remaining - len(selected_other)
            selected_other = selected_other + extra_hard[:fill]
        benign = selected_hard + selected_other
        random.shuffle(benign)

    final_train = attacks + benign
    random.shuffle(final_train)
    random.shuffle(eval_pool)

    # ── Write outputs ─────────────────────────────────────────────────────────
    _write_jsonl(TRAIN_FILE, final_train)
    _write_jsonl(EVAL_FILE,  eval_pool)

    # ── Dataset provenance sidecar (zeek_dataset.meta.json) ────────────────────
    # Records git SHA, CLI args, the resolved preprocess knobs, counts, and a
    # content hash — so an adapter trained on this file can prove what produced it.
    try:
        from ids import preprocess_config as pc
        from ids.run_manifest import write_dataset_meta
        write_dataset_meta(
            TRAIN_FILE, EVAL_FILE,
            args={"ctu_only": args.ctu_only, "no_uwf": args.no_uwf,
                  "no_reason": args.no_reason},
            config={
                "reason":                    os.environ.get("IDS_NO_REASON") != "1",
                "TRAINING_FACTOR":           pc.TRAINING_FACTOR,
                "FINAL_ATTACK":              pc.FINAL_ATTACK,
                "FINAL_BENIGN":              pc.FINAL_BENIGN,
                "EVAL_FRAC":                 pc.EVAL_FRAC,
                "RANDOM_SEED":               pc.RANDOM_SEED,
                "CONN_STATE_MASK_PROB":      pc.CONN_STATE_MASK_PROB,
                "CONTEXT_MASK_PROB":         pc.CONTEXT_MASK_PROB,
                "HARD_BENIGN_MIN_SCORE":     pc.HARD_BENIGN_MIN_SCORE,
                "HARD_BENIGN_TARGET_FRAC":   pc.HARD_BENIGN_TARGET_FRAC,
                "CTU_MALWARE_ATTACK_BUDGET": pc.CTU_MALWARE_ATTACK_BUDGET,
            },
            counts={"train": len(final_train), "eval": len(eval_pool),
                    "attacks": len(attacks), "benign": len(benign)},
            sources=Counter(s["source"] for s in final_train),
        )
        print(f"   📋 Dataset meta → {os.path.splitext(TRAIN_FILE)[0]}.meta.json")
    except Exception as e:
        print(f"[WARN] could not write dataset meta: {e}")

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

    # ── Attack composition sanity check ──────────────────────────────────────
    # Warn if any single source dominates the attack pool.  At TRAINING_FACTOR=0.5
    # UWF reached 44.7% of attacks because the other pools were halved while
    # FINAL_ATTACK stayed at 120k — causing the model to over-learn short SF TCP.
    ATTACK_SOURCE_WARN_PCT = 25.0
    n_atk_total = max(len(attacks), 1)
    atk_by_source = Counter(s["source"] for s in attacks)
    warnings_found = False
    for src, n_src in atk_by_source.most_common():
        pct = 100 * n_src / n_atk_total
        if pct > ATTACK_SOURCE_WARN_PCT:
            print(f"\n⚠  ATTACK COMPOSITION WARNING: '{src}' = {pct:.1f}% of attacks "
                  f"({n_src:,} / {n_atk_total:,}) — exceeds {ATTACK_SOURCE_WARN_PCT:.0f}% threshold.")
            print(f"   Consider lowering TRAINING_FACTOR or raising the cap for other sources.")
            warnings_found = True
    if not warnings_found:
        top_src, top_n = atk_by_source.most_common(1)[0]
        print(f"\n✅ Attack composition OK: top source '{top_src}' = "
              f"{100*top_n/n_atk_total:.1f}% (threshold {ATTACK_SOURCE_WARN_PCT:.0f}%)")
