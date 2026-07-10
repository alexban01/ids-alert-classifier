# IDS Alert Classifier — Current State

> **THIS IS THE SINGLE SOURCE OF TRUTH.** Update *only* this file (`/home/alex/fine_tunning/STATE.md`).
> The `ids-project` skill's `references/current-state.md` is a **symlink** to this file, and the
> auto-memory `project_ids_classifier.md` is just a pointer here. Do not maintain a second copy.

_Last updated: 2026-07-05 (v12.2 data-volume check trained + benchmarked: MCC +0.731,
clearly beats v12.1's +0.669 on half the data → **downsampled-data gate PASSED**, use
`zeek_dataset_50pct.jsonl` for v13.x. ⚠️ Confound: v12.2 accidentally trained at
r=16/α=32 — train.py had already been flipped to the v13 setting — so it's really
"v13 on 50% data", not a pure volume ablation vs v12.1's r=32/α=64. See Next steps)_

---

## ⚠️ STATUS (2026-07-02) — benchmark fixed + re-baselined; V12/v12.1 still need a clean retrain

**Benchmark methodology bugs fixed** (landed directly on master 2026-07-02 — a
one-off exception to the usual Codex-on-`development` workflow, at the user's
request, to unblock a same-day clean number): CTU-13 `conn_state` is now mapped to
Zeek equivalents before scoring (was raw unmapped Argus tokens), and all 7 loaders
switched from head-of-file bucket filling to seeded reservoir sampling over the full
source (was: first-300-in-file-order). Verified: Win7AD-1's attack sample composition
is now 266/34 Human/Trickbot (≈89%/11%, matching the capture) vs the old 79/221
(inverted). Cross-run determinism confirmed (two runs, identical cache hash).

**Re-baselined 4-way benchmark (2026-07-02, `--regen`, corrected methodology):**

| Model | MCC | Atk Recall | FP Recall | Win7AD-1 Atk |
|---|---|---|---|---|
| v11 ckpt-11313 (ep1) | **+0.777** | 81.1% | 95.8% | 83.0% |
| v11 ckpt-22626 (ep2) | +0.770 | 80.1% | 96.0% | 74.3% |
| v12 ckpt-10000 (ep1) | +0.732 | 77.9% | 94.3% | 73.0% |
| v12.1 ckpt-10000 (ep1) | +0.669 | 74.1% | 91.8% | 52.7% |
| v12.2 adapter (ep1, 50% data, **r=16**) | +0.731 | 84.2% | 88.8% | 79.7% |

(v12.2 row added 2026-07-05 — complete 1-epoch run on `zeek_dataset_50pct.jsonl`,
same cache/samples as the 2026-07-02 rows. Highest attack recall of the four, but
lowest benign recall — IoT-23 FP recall 58.7% is the drag. Kelihos 49.3% (best),
Echo atk 89.0%, Trickbot 15% (5/34). eval_loss 0.194, 2.1 h on the 3070.)

Full report: `results/benchmark_realworld_report.txt`. Note the sampling fix alone
moved V11 ep1's Win7AD-1 attack recall from 73.0% (old buggy benchmark) to **83.0%**
— materially closer to V10's 87.1% target than previously measured; some of the
"V11 regressed on Win7AD-1" framing lower in this file may itself be a sampling
artifact from the pre-fix benchmark, not a pure training effect. Worth revisiting in
the thesis writeup rather than assumed.

**V12/v12.1 still underperform V11 ep1 even under the corrected benchmark** — so
this is no longer a pure measurement artifact. But it still isn't a clean verdict on
the V11→V12 composition fixes, because the model/training side remains confounded:
- Both runs were interrupted at **step 10000 of ~19980** (just past the epoch-1
  boundary of a planned 2-epoch run) — **neither completed**.
- **The planned reason-mode A/B never happened.** Plan (2026-06-30) was V12 =
  no-reason, v12.1 = with-reason comparison arm. v12.1's `run.json` confirms
  `"reason": false` — identical to v12. v12.1 is a second no-reason attempt, not the
  with-reason arm. That ablation is still outstanding (see Next steps).
- **v12's training data is unrecoverable.** `zeek_dataset.jsonl` was regenerated
  (2026-06-30T21:13:40) *after* v12's checkpoint-10000 was saved (20:32:34),
  overwriting whatever v12 actually trained on (`run.json`: `matches_meta: false`,
  `train_sha256: null`). v12.1 trained on the current on-disk dataset (hash
  confirmed match). v12's exact composition can no longer be audited.
- **v12 vs v12.1 diverge more than expected for "the same config"**: both are
  nominally reason-off/factor=1.0/1-epoch, yet v12.1 is substantially worse across
  nearly every source (Win7AD-1 atk 73.0%→52.7%, IoT-23 FP 82.7%→68.3%). Likely
  reflects high run-to-run variance from a single incomplete epoch plus two
  different random dataset regenerations — but can't be fully explained given v12's
  lost provenance. A data point for why a clean, complete, provenance-tracked run
  matters, not just a nice-to-have.

**Bottom line:** V11 ckpt-11313 (ep1) remains the best validated candidate. Whether
the V11→V12 composition fix (factor=1.0, UWF cap, CTU-Malware budget) actually helps
is still an open question — answer it with one clean retrain, not these two.

---

## Active version: **V11 ckpt-11313 (ep1)** — shipped/best model; V13 (r=16) in prep

V11 ckpt-11313 (ep1) is the latest fully-trained, benchmark-validated, best-on-every-metric
model — treat as current until V13 says otherwise. V12/v12.1's `factor=1.0` composition
retrain is **not** planned (see Next steps #2); instead the active experiment is **V13**
(LoRA r=16, see Next steps #3), targeting the same Win7AD-1/OOD gap via a different,
better-evidenced lever (V10 already hit 87.1% at r=16; V11/V12's r=32 never matched it).

## Prior version: **V11 trained (regressed) → V12 = next + final thesis model**

**Real objective (user, 2026-06-26):** improve ATTACK recall without materially hurting
benign precision. NOT fixated on a specific CTU-Malware share / 40% number — that was a
proxy. The binding problem is *coverage* of attack types the model misses on OOD.

**Latest trained model = V11** (`models/v11-ids-model/`, 2 LoRA checkpoints):
- `checkpoint-11313` (epoch 1, eval_loss 0.1608) — **better on benchmark: MCC +0.866,
  Atk 88.0%, FP 98.2%**
- `checkpoint-22626` (epoch 2, eval_loss 0.1570) — `best_model_checkpoint` by eval_loss,
  but **worse on benchmark: MCC +0.820, Atk 85.4%, FP 96.1%** (ep2 overfits — eval_loss is
  misleading here; ep1 generalizes better)
- **V11 regressed on Win7AD-1**: trained at `TRAINING_FACTOR=0.5` by accident (RTX 5090 OOM,
  thesis_notes_10 §4). UWF reached **44.7%** of attacks (31.8% even at factor=1.0) — its
  halved pool stayed large vs shrunken IoT-23/CTU-13 pools AND every UWF (SF) sample carries
  a 2× draw weight. Result: Win7AD-1 attack recall **−29.3 pp**, undercutting V10's 87.1%.

**V12 = the factor=1.0 fix, designated final thesis model.** All V12 preprocessing fixes
were in config (verified) and V12 *was* attempted — twice, as v12 and v12.1 — but both
runs were interrupted at step 10000 of ~19980 (just past epoch 1 of 2) and neither
produced a usable, provenance-clean result. See the ⚠️ STATUS block at the top of this
file for the full diagnosis. A clean, uninterrupted retrain is still needed.

**Status / unblocked this session (2026-06-30, pre-training snapshot):** the following
was true before training started that day; superseded by the ⚠️ STATUS block above for
what actually happened once training ran.
- Shared-parsing/inference refactor merged to master (`zeek_log_utils.py` + `infer_utils.py`,
  ~1.7k lines deduped; training cost ~halved: packing + 2 epochs + eval subsample). Master is
  ahead of origin (unpushed).
- **IoT-23 restored** (was deleted — would have silently broken V12). Re-extracted from
  `~/Downloads/iot_23_datasets_small.tar.gz` (verified: Stratosphere "small" labeled conn.log,
  21-field TSV, Malicious/Benign). Loader reads ≤8000 rows/file, so each file truncated to 10k
  lines → `datasets/iot-23/` is 24 MB (not 43.8 GB), loader output byte-identical. Loader
  verified: **45,109 attacks / 10,353 benign** across 23 files.
- **Benchmark repointed to V11**: `benchmark_realworld.py` MODELS now runs only the two V11
  checkpoints (v9.1 commented out). Uses `models/v11-ids-model/checkpoint-{11313,22626}`.
- **V12 training data regenerated** (factor=1.0, IoT-23 restored) → `zeek_dataset.jsonl`
  (360k: 120k atk / 240k ben). Composition verified healthy: UWF clamped to exactly **25.0%**
  (was 31.8% in V11), no source dominance (iot23 24%, uwf 25%, ctu13 24%, ctu_malware 20%,
  unsw 7%), 120k attack target fully met. Hard benigns = 81.9% of benign train. Ready to train V12.

---

## Version timeline & Win7AD-1 recall (the key story)

The v9.1 "LR beats LLM 35×" result was **superseded by V10** — the OOD gap was a training
*composition* problem, not a fundamental LLM-OOD limit. Once trained with proper composition,
the LLM matches LR on the hard OOD probe.

| Ver | LoRA | CTU-Mal scenarios | Win7AD-1 attack recall | Note |
|---|---|---|---|---|
| v9.1 | r=32 | — | **11.3%** (LR 78%) | the original "publishable" gap |
| **V10** | **r=16/α=32** | 10 | **87.1%** (benign 84.4%) | **fixed it** — LLM ≈ LR (LR ~88.7%) |
| V11 | r=32/α=64 | 15 | regressed −29.3 pp | factor=0.5 UWF-bias accident |
| V12 | r=32/α=64 | 15 | target ~84–87% (recover) | factor=1.0 fix; final thesis model |

**V11 benchmark (thesis_notes_10) — overall:** ep1 (ckpt-11313) **MCC +0.866** / Atk 88.0% /
FP 98.2%; ep2 (ckpt-22626) MCC +0.820 / Atk 85.4% / FP 96.1%. IoT-23 FP recall **+30.7 pp**
(hard-benign fix helped), Echo attack recall +9.0 pp — but Win7AD-1 attack recall −29.3 pp.

**V11 OOD-only benchmark (2026-06-26 local re-run, `--ood`, ep1 vs ep2):**
- **ep1 wins decisively** (OOD MCC +0.699 / Atk 76.4% / FP 94.8% vs ep2 +0.600 / 71.1% / 90.0%).
  Confirms eval_loss mis-selects: ep2 overfits. ⚠️ **V12 caveat:** `load_best_model_at_end`
  saves ep2 as `models/v12-ids-lora-adapter` → DO NOT deploy it; benchmark both V12 checkpoints and
  ship the ep1-equivalent.
- Per source (ep1): Win7AD-1 73.0% atk / 93.0% ben; Echo 75.7% / 96.7%; Kelihos 80.7%.
- **SUPERSEDED BY TASK-012 (2026-07-02): the 79/221 Human_attacks/Trickbot split below was the**
  **pre-fix sampling-bug composition — inverted.** The corrected reservoir-sampled Win7AD-1 is
  **266 Human_attacks / 34 Trickbot** (≈89%/11%, matching the actual capture). So Human_attacks,
  not Trickbot, sets the probe score: at current V11 ep1 recall (Human_attacks 238/266=89.5%,
  Trickbot 11/34=32.4%), lifting Trickbot to 100% only buys +7.7pp (83.0%→90.7%); a 5pp gain on
  Human_attacks buys roughly the same. Effort should target lateral-movement/REJ-RSTR pattern
  coverage, not Trickbot specifically. Original (now-stale) paragraph kept below for history:
  Win7AD's 300 attacks = 79 Human_attacks (lateral REJ/RSTR) + 221 Trickbot C2. ep1: **Human_attacks
  94%** (was 13% in v9.1!), **Trickbot C2 only 66%** (145/221). Trickbot is the majority of Win7AD
  (221/300), so it alone sets the probe score: lifting it 66%→~85% → Win7AD ~87% (V10 level).
  **OOD framing (user, correct):** Trickbot is NOT a training family + no synthetic data, so 66%
  recall on a fully unseen family is a *positive* generalization result, not a failure — it only
  reads as a "gap" vs V10's 87% on the same probe. **DECISION DEFERRED:** do NOT decide on
  targeted Trickbot coverage (RSTO→4134/22299) until **after V12 is trained + benchmarked** —
  V12's composition fix may already move it; add coverage only if it falls short.
- **Kelihos 80.7% was itself a sampling artifact — REVISED to ~40-44%** (2026-07-02,
  TASK-012 reservoir-sampling fix). The old benchmark always drew the same first 300
  lines of a 371,852-flow capture; that head slice is missing 9 of 13 conn_states
  present in the full file (RSTR/RSTRH/SHR/S1/RSTOS0/S2/REJ/SH/S3 — all absent from
  the first 300, all present at low frequency across the full capture) and is skewed
  toward the two easy majority states (S0/SF). A fair reservoir sample scores ~40-44%
  across all 4 models benchmarked 2026-07-02 (v11 ep1 44.0%, ep2 39.0%, v12 39.3%,
  v12.1 43.0%) — much closer to the *original* "structural floor" framing than the
  80.7% revision was. Neither "~0%" nor "~80%" is right; ~40% on a fully OOD P2P-spam
  family is the current honest number. See `notes/thesis_notes_13.txt`.
- Benigns safe (FP recall 93-97% on ep1). "Don't hurt benigns" constraint met.
- Full write-up: **`notes/thesis_notes_12.txt`** (this section); **`notes/thesis_notes_13.txt`**
  for the benchmark-methodology fix and its effect on these numbers.

**Host Pass-2 (host-level aggregation): GATED behind `--host-pass2`, off by default.** Null
result — aggregating per-flow preds to a host verdict (ATTACK if any flow is attack) gives MCC
+0.029 (ep1) / +0.136 (ep2) ≈ random, FP recall 33% (flags 2/3 benign hosts). Code kept for
reproducibility; documented as a tried-and-failed approach in thesis_notes_12.

**Baseline (factor=1.0 era, thesis_notes_10):** LR overall MCC +0.709; on Win7AD-1 LR
MCC **+0.845** (Atk 88.7%, FP 95.7%) — LR's strongest probe; RF MCC +0.622 (Atk 59.3%,
lateral movement 0%, Trickbot 81%). LR's weakness: IoT-23 benign/FP recall only 42.4%.

---

## Architecture (V11 / V12)

- Base: `Qwen/Qwen2.5-1.5B-Instruct`; LoRA **r=32, alpha=64**, 7 target modules (all attn + MLP)
  - NOTE: **V10 used r=16/α=32**; V11 raised to r=32/α=64 (more capacity for residual confusion).
- **max_length: 512** (train.py default; token lengths mean 296 / p99 500)
- Prompt: 10-field Zeek-native + Service + Dest Port + optional [HTTP/DNS/SSL]
- Context masking: `CONN_STATE_MASK_PROB=0.20`, `CONTEXT_MASK_PROB=0.50`
- [BEHAVIOR] section **present** (~40% coverage in V12 data: atk 38.6% / ben 41.6%) — earlier
  note that it was "dropped" was wrong; loaders still build behavior contexts. Host Pass-2 retired.

---

## V11 → V12 changes (design — attempted twice, both interrupted; see ⚠️ STATUS block)

1. **`TRAINING_FACTOR`: 0.5 → 1.0** — the core fix; restores per-source pool sizes.
2. **`UWF_ATTACK_CAP = 25_000`** — bounds UWF before the weighted draw (it had hit 31.8–44.7%).
3. **Per-source 25% draw cap** in `preprocess_zeek.py` — trims any source >25% of FINAL_ATTACK
   after the weighted draw, refills from sources still under cap.
4. **`CTU_MALWARE_ATTACK_BUDGET = 48_000`** — reserves CTU-Malware before the draw. Inert:
   actual pool **23,918** (15 scenarios, factor=1.0), sampler takes `min(budget,pool)`, so all
   ~24k taken and 24k≡48k (pool just under 24k — barely). CTU-Malware = ~20% of attacks. 40%
   share is NOT reachable by capping IoT-23/CTU-13 (UNSW + UWF alone exceed the needed
   denominator); only lowering FINAL_ATTACK or sourcing more CTU-Malware data would. Given the
   reframed objective, target missed attack types, not a share number.
5. **Reason pools expanded**: 77 attack / 48 benign reasons (added Murlo DCE/RPC, Rbot-v2 ICMP
   sweep, + benign complements).
6. SF/S1/OTH attacks keep 2× draw weight (recall on completed-connection C2/Credential Access).

**V12 expected (thesis_notes_11 §4):** Win7AD-1 recall ~84–87% (recover to V10 level), Echo
≥75%, Kelihos ~81–83%, IoT-23 FP ~99%, overall MCC +0.88–0.90. **Note (2026-07-02):** the
Kelihos ~81–83% target was itself derived from the since-revised 80.7%/83.3% number (see
Section "Kelihos 80.7% was itself a sampling artifact" above) — ~40-44% is now the
honest baseline to compare against, not ~81–83%.

---

## Training configs (corrected 2026-06-26)

**RunPod RTX 5090 (32 GB) — full training:** `batch=24, grad_accum=1 (eff=24),
gradient_checkpointing=False, max_length=512`
**Local RTX 3070 (8 GB) — validation only:** `batch=4, grad_accum=6 (eff=24),
gradient_checkpointing=True, max_length=512, pin_memory=False, num_workers=0`
**Shared:** **2 epochs**, **packing=True**, lr=2e-4, cosine_with_restarts (cycles=epochs),
warmup=0.03, weight_decay=0.01, load_best_model_at_end (eval_loss). `--eval-subset` trims eval.

**Stop & resume (local 3070 path, added 2026-06-28):** `train.py --save-steps N` switches
save+eval from per-epoch to every `N` optimizer steps (eval_steps pinned = save_steps, since
load_best needs matching strategies), so a run can be killed mid-epoch and continued. `0`
(default) = unchanged epoch behaviour; don't pass on RunPod. `train.py --resume` continues from
the latest checkpoint in `OUTPUT_DIR` (`--resume <path>` for a specific one) — restores model +
optimizer + LR scheduler + step counter; warns & starts fresh if none found. Dataset/batch/
packing/epochs must be unchanged (HF fast-forwards the dataloader on the same ordering).
`save_total_limit=2` unchanged → most-recent stays resumable. Usage:
`train.py --save-steps 500` then `train.py --save-steps 500 --resume`. (RunPod runs are short
enough to not need this, and pod checkpoints would need a persistent volume to survive a stop.)

---

## Experiment tracking & provenance (added 2026-06-28)

`ids/run_manifest.py` (stdlib-only) auto-stamps every run so adapters aren't anonymous:
- **`zeek_dataset.meta.json`** — written by `preprocess_zeek.py`: git SHA, CLI args, resolved
  preprocess knobs (incl. `reason` on/off, TRAINING_FACTOR, masking probs), counts, content hash.
- **`models/<adapter>/run.json`** — written by `train.py`: hyperparams + dataset link (by content
  hash; surfaces `reason`/TRAINING_FACTOR from the meta) + best `eval_loss` + runtime. Travels with
  the adapter (gitignored).
- **Known gap (found 2026-07-02, v12 incident):** `run.json` is only written when `train.py`
  finishes/exits normally. If a run is interrupted (killed, crashed) *and* `zeek_dataset.jsonl` is
  regenerated before that run's `run.json` is reconstructed, the dataset hash/content the
  interrupted run actually trained on is unrecoverable — `matches_meta` comes back `false` and
  `train_sha256` is `null` forever. Rule: don't re-run `preprocess_zeek.py` while a checkpointed
  run is still in flight without archiving a copy of the `zeek_dataset.jsonl` + `.meta.json` it used.
- **`EXPERIMENTS.md`** (repo root, committed) — generated leaderboard, one row/run: settings +
  eval_loss + MCC/recalls. `benchmark_realworld.py` writes MCC back into `run.json` (FULL mode only),
  then regenerates. Rebuild manually: `.venv/bin/python scripts/experiments.py`.
- **Known gap (found 2026-07-02):** `regenerate_experiments_md()` only scans one level —
  `models/<name>/run.json` — via `os.listdir(MODELS_DIR)`. It silently misses `run.json`
  nested under a checkpoint dir (`models/<name>/checkpoint-N/run.json`), which is exactly
  where it lives for any adapter that never finished training into a top-level
  `<name>-ids-lora-adapter/` dir (v12, v12.1 — see ⚠️ STATUS). Confirmed 2026-07-02: v12/v12.1
  `run.json` both have a real `benchmark.mcc`, but `EXPERIMENTS.md` still printed
  "(no runs yet)" after a manual rebuild. Not fixed — flagged as a REVIEW_TASKS.md follow-up
  (TASK-017); until then, read MCC straight from `models/<name>/checkpoint-N/run.json` or the
  benchmark report, not from `EXPERIMENTS.md`, for any interrupted/checkpoint-only run.

**REASON ablation:** `preprocess_zeek.py --no-reason` drops the (randomly-picked, non-grounded)
REASON line → targets become bare `VERDICT: <X>` with `SYSTEM_PROMPT_VERDICT_ONLY`. Measured ~11–14%
fewer training tokens (mean seq 303→260) → proportional savings with packing. Compare runs by **MCC**,
not eval_loss (full-sequence loss, different target token counts → not cross-run comparable; the
`EXPERIMENTS.md` eval-loss column is footnoted as within-run-only).

**Prompt matching is automatic on all HF inference paths:** `resolve_system_prompt()`
(`ids/infer_utils.py`) reads the adapter's `run.json` and serves the verdict-only prompt when
`dataset.reason == False` (no `run.json` → default). Wired into `benchmark_realworld.py` (prints a
provenance banner), `benchmark_v6.py`, `scripts/classify_conn_log.py`, `scripts/classify_weird_log.py`.
`train.py` records `dataset.reason` by **sniffing the dataset** (`detect_reason_from_dataset`) — correct
on RunPod where `zeek_dataset.meta.json` isn't uploaded (was a silent-mismatch bug: meta-only detection
would have served the reason-on prompt to v12). **Ollama/GGUF has no run.json:** pass `--verdict-only`
to `benchmark_ollama.py` / `classify_conn_log.py --ollama`; `Modelfile` documents the verdict-only SYSTEM line.

---

## OOD probe configuration

| Probe | Role | Notes |
|---|---|---|
| CTU-SME-11 (Windows7AD-1) | **Primary OOD** | Outbound lateral movement (REJ/RSTR→AD ports) + Trickbot C2 |
| CTU-SME-11 (Amazon Echo) | Easy OOD | IoT-context scans |
| CTU-Malware Botnet-3 (Kelihos) | Hard floor | P2P spam, ~40-44% recall under fair sampling (2026-07-02) — hard but not undetectable |

Benchmark sources (`ALL_SOURCES`): iot23, ctu13, uwf, ctu_normal + ctu_win7ad, ctu_sme11,
ctu_botnet3. 300 samples/(source,class). Cache: `results/benchmark_realworld_cache.json`.

---

## CTU-Malware training scenarios (15 active in `CTU_MALWARE_SCENARIOS`)

Botnet-42 (Ramnit), 43 (Neris), 44 (Ngrbot), 45 (Rbot), 46 (Virut), 48 (Sogou), 52 (Htbot),
53 (NSIS.ay), 54 (Siemens), 78-2 (Zeus), + V11 additions: 25-1 (Zbot), 47 (DonBot), 49 (Murlo),
50 (Neris-v2), 51 (Rbot-v2). (25-2/55/61-1/64 excluded — empty Label column.)
**Permanently held out:** Botnet-3 (Kelihos), SME-11 (Echo + Windows7AD-1).

---

## Next steps

1. ~~Land TASK-011/012, regen, re-baseline V11~~ — **done 2026-07-02**, see ⚠️ STATUS
   block. Corrected numbers are now in `results/benchmark_realworld_report.txt`.
2. ~~Retrain V12 cleanly~~ — **superseded 2026-07-02.** V12/v12.1 already underperform
   V11 ep1 on every checkable metric (Win7AD-1, IoT-23 FP, Trickbot, MCC) even under the
   fixed benchmark, and the original motivation (Win7AD-1 "regressed −29.3pp") turned out
   to be largely a benchmark-sampling artifact — V11 ep1 is actually at 83.0% Win7AD-1,
   only 4.1pp off V10's 87.1%. Chasing the `factor=1.0` composition fix further isn't
   worth another retrain by default; **V11 ep1 is the current shipped/best model.**
3. ~~v12.2 data-volume check~~ — **done 2026-07-05: gate PASSED.** Trained
   (complete, provenance-clean, 2.1 h local) + benchmarked: **MCC +0.731** vs v12.1's
   +0.669 on half the data — no drop, clear improvement → v13.x runs use
   `--dataset zeek_dataset_50pct.jsonl`. **Confound:** train.py was already at
   r=16/α=32 (v13 setting) when v12.2 ran, so it changed rank AND volume vs v12.1
   (r=32, full data). Since it *beat* v12.1 anyway, both changes are jointly
   validated as at-least-not-worse; v12.2 effectively IS "v13 packed, 50% data,
   1 epoch" — treat it as the v13 baseline arm. Still 4.6 pp MCC below V11 ep1
   (+0.777), driven by IoT-23 benign recall (58.7%). Original plan text follows.
   **v12.2 data-volume check (2026-07-03, gates v13 — run this first).** The training-token
   count for the current dataset was measured directly (92.9M train / 11.9M eval tokens,
   360k/46.2k samples) and v12.1's own loss curve shows clear saturation well inside epoch 1
   (train loss flat ~0.185 from step ~5000 on; eval loss decelerating from −0.022/500-steps
   early to −0.001/500-steps by step 10000) — evidence the run isn't data-starved. Testing
   whether that means training can go faster: `preprocess_downsample.py` takes a
   class-stratified (not source-stratified — source labels aren't retained in the final
   messages-only JSONL) random subsample of `zeek_dataset.jsonl` at a fixed seed.
   `zeek_dataset_50pct.jsonl` (180k, sha256 `18c75dd005579c67a341c594b2b9f2e642514b52187a03768f1db6c6a8e79aba`)
   already generated. **Do NOT use `TRAINING_FACTOR` for this** — it doesn't scale
   `FINAL_ATTACK`/`FINAL_BENIGN` (fixed at 120k/240k), only per-source pool caps, so lowering
   it re-skews source composition (the exact mechanism behind V11's Win7AD-1 regression)
   without even reducing final dataset size. Downsampling the already-composed final file
   avoids that.
   - Same r=32/α=64/packing=True as v12.1 — only data volume changes. No `--save-steps`, so
     epoch-based checkpointing lands on a true epoch-1 boundary. Compare v12.2's epoch-1
     checkpoint against v12.1's ckpt-10000 **by epoch, not by step count** — the downsampled
     set has ~half the steps/epoch, so v12.2's epoch-1 checkpoint lands at a much lower
     absolute step, which is expected and correct.
   - **Commands (local RTX 3070):**
     ```bash
     .venv/bin/python train.py --dataset zeek_dataset_50pct.jsonl --tag v12.2
     ```
     Benchmark with `.venv/bin/python benchmarks/benchmark_realworld.py --regen` (update
     `MODELS` to point at the new checkpoint dir first) and compare against v12.1's numbers
     already in `results/benchmark_realworld_report.txt`.
   - **If v12.2 ≈ v12.1 (no real MCC drop): switch to downsampled data for v13/v13.1/v13.2**
     (~2x faster local runs — swap `--dataset zeek_dataset_50pct.jsonl` into the v13 commands
     below). **If v12.2 is clearly worse: run v13 on the full dataset** as originally planned.
4. **V13 (2026-07-02, local RTX 3070, `train.py --tag v13`): LoRA r=16/α=32**
   (reverted from V11/V12's r=32/α=64, back to V10's setting — V10 is still the
   best-ever Win7AD-1 result and was never matched at r=32; higher rank may just add
   ID-memorization capacity without OOD benefit, same shape as the epoch-2-hurts finding).
   Plan (user's, 2026-07-02):
   - **v13**: r=16, packing ON (default), presumably 1 epoch.
   - **v13.1**: same as v13 but `--no-pack`. Motivated by a real finding, not just caution:
     TRL 0.29.1's default `packing_strategy="bfd"` needs a flash-attention variant to stop
     packed samples from attending to each other; this repo has no `flash-attn` installed
     and never sets `attn_implementation` (defaults to SDPA/eager) — so packed sequences
     (~1.46 samples/512-token slot on average) likely DO leak cross-sample attention. Loss
     *masking* is unaffected (full-sequence loss either way, no completion-only masking),
     but that's a narrower claim than "packing doesn't change the objective." See the
     `--no-pack` help text in `train.py` for the full explanation.
   - **v13.2**: 2-epoch run of whichever of v13/v13.1 wins — tests whether the "epoch 2
     overfits and hurts OOD" finding (established at r=32 on V11) still holds at r=16;
     lower-capacity adapters may not overfit as fast.
   - `train.py` now takes `--tag <name>` (default `v13`) so each variant writes to its own
     `models/<tag>-ids-model` / `models/<tag>-ids-lora-adapter` without clobbering the others.
   - **No `preprocess_zeek.py` rerun needed** — `zeek_dataset.jsonl` on disk (sha256
     `afbe57b5e0…`, unchanged since 2026-06-30) is byte-identical to what v12.1 trained
     on (`run.json` confirms the same hash). Reusing it holds composition constant so
     v13 isolates the rank/packing variables cleanly. Only rerun preprocessing if you
     want to change composition, not for these runs.
   - **Commands (local RTX 3070) — updated 2026-07-05: gate passed, use 50% data;
     the packed 1-epoch baseline arm is already covered by v12.2 (MCC +0.731):**
     ```bash
     # v13 baseline (r=16, packed, 1 ep, 50% data) == v12.2 — already done, +0.731
     .venv/bin/python train.py --tag v13.1 --epochs 1 --no-pack --dataset zeek_dataset_50pct.jsonl
     # then whichever of v12.2/v13.1 wins on MCC:
     .venv/bin/python train.py --tag v13.2 --epochs 2 --dataset zeek_dataset_50pct.jsonl
     ```
     Benchmark each with `.venv/bin/python benchmarks/benchmark_realworld.py --regen`
     (update `MODELS` in that script to point at the new checkpoint dirs first).
5. **Reason A/B still outstanding** (unrelated to v13, lower priority): the with-reason
   arm never happened (v12.1 turned out reason-off, same as v12). Revisit after v13.
6. Then research recall-coverage additions for any attack types still missed (see the
   corrected Human_attacks vs Trickbot framing above — target lateral-movement/REJ-RSTR
   coverage over Trickbot specifically, now that the true composition is known).

---

## Future work

- **Make the REASON field actually work (grounded explanations).** Reasons are currently picked
  at random from pools (`pick_reason()`), not derived from input features, and the prompt emits
  `VERDICT:` *before* `REASON:` — so the reason can't aid verdict accuracy (no CoT benefit) and at
  inference the model fabricates plausible-but-ungrounded justifications. Two honest options:
  (1) drop it entirely (`--no-reason`, verdict-only) — cleaner classifier, ~11–14% cheaper; or
  (2) **template the reason from actual features** (e.g. "S0 state + high orig-pkt count → scan
  attempt") so explanations are evidence-tied. Option 2 is the only one that adds real value
  (interpretable IDS alerts — a defensible thesis contribution). The random-reason status quo is
  the worst of both (pays tokens, dilutes loss, ships hallucinated rationales).
  - **Not actually started:** the "in progress" v12/v12.1 pair turned out to be two
    no-reason attempts (both interrupted), not a no-reason/with-reason A/B — see ⚠️
    STATUS block, top of file. Still queued; see Next steps #3.
- **Qwen2.5-3B capacity ablation** — queued control (see Deferred below).

**Deferred (decided 2026-06-26):** base-model swap is NOT the recall lever — V10 already hit
87% Win7AD on the 1.5B, so capacity isn't the bottleneck (data composition is). A
**Qwen2.5-3B ablation** (trivial swap — same tokenizer/template) is queued as a *control* to
confirm "capacity is not the bottleneck," to run **only after V12 is trained + benchmarked**.
Do not swap the final thesis model (v4→v12 all Qwen2.5-1.5B; edge-deploy favors small). 7B
skipped (borderline on 3070 8 GB). KDE-off (~900 MB) not needed for model size — only useful
to raise benchmark BATCH_SIZE.

---

## Repo layout (restructured 2026-06-26)

Shared library code is the **`ids/` package** (import as `from ids.<module> import …`);
`ids/loaders/` holds the dataset loaders. Entry-point scripts (`preprocess_zeek.py`,
`train.py`) stay at root and run from root; `benchmarks/`, `scripts/`, `tests/` add the
root to `sys.path` so `import ids` resolves. All model outputs (checkpoints, adapters,
merged, `*.gguf`) live under **`models/`** (gitignored) — e.g.
`models/v11-ids-model/checkpoint-11313`, `models/v12-ids-lora-adapter`. Training jsonls
stay at root (train CWD + RunPod upload convention).

## Key files

| File | Purpose |
|---|---|
| `preprocess_zeek.py` (root) / `ids/preprocess_config.py` | Dataset builder; TRAINING_FACTOR, caps, scenarios, reason pools |
| `ids/prompt_utils.py` | `build_prompt()`, `extract_verdict()`, `extract_reason()` |
| `ids/infer_utils.py` / `ids/zeek_log_utils.py` | Shared model load + chat templating; shared Zeek TSV parsing |
| `ids/preprocess_sample.py` / `ids/behavior_features.py` | `make_sample()`, hard-benign scoring; behavior context features |
| `ids/loaders/` | Per-source dataset loaders (imported by `preprocess_zeek.py`) |
| `train.py` (root) | QLoRA SFTTrainer (2 epochs, packing, max_length 512); writes `models/v12-ids-*` |
| `benchmarks/benchmark_realworld.py` / `benchmarks/bench_loaders.py` | Primary benchmark (MODELS→`models/v11-…`); data loaders |
| `scripts/analyze_gap.py` / `scripts/baseline_ml.py` | conn_state/port gap analysis; RF+LR baseline |
| `scripts/merge_adapter.py` | Merge LoRA adapter for GGUF conversion |
| `ids/run_manifest.py` / `scripts/experiments.py` | Run provenance (meta/run.json sidecars + `EXPERIMENTS.md` ledger); manual rebuild |

---

## Python envs

- Project: `.venv/bin/python` / `.venv/bin/pip`
- GGUF conversion only: `llama.cpp/.venv/bin/python`
- Never use system `python3` / `pip3`
