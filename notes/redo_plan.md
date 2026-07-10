# REDO PLAN — "v2 of the project": rewrite + method redesign + thesis-first rigor

Status: **plan only** (agreed 2026-07-05). To be revised after v13/v13.1/v13.2 results.
Fixed constraints: base model stays `Qwen2.5-1.5B-Instruct`; budget stays local RTX 3070
+ occasional RunPod 5090 (~$0.44/hr, a handful of full runs). Datasets and OOD probes
are allowed to change. REASON gets **grounded** (feature-templated), not random.

---

## 0. Why redo, in one paragraph

Eleven versions produced three genuinely good findings (composition drives OOD; the
"confidently wrong" REASON; LLM non-degenerate where RF/LR are degenerate) — but every
headline number has been re-litigated at least once because the *measurement* and
*provenance* machinery was retrofitted, not designed in: benchmark sampled file heads
(thesis_notes_13), fed unmapped Argus states, v12's training data is unrecoverable,
eval_loss mis-selects checkpoints, the REASON is random noise the model learns to
hallucinate, and composition emerged from four interacting mechanisms (caps ×
TRAINING_FACTOR × 2× SF weights × post-draw trims) that produced the v11 accident.
The redo bakes each lesson in structurally so the failure class is impossible, not
just patched.

---

## 1. Design principles (the distilled lessons)

1. **Provenance is written first, not on exit.** A run manifest is written *before*
   training starts (dataset hash, resolved config, git SHA + diff hash) and updated at
   every checkpoint save. An interrupted run is never unauditable again (v12 lesson).
   The exact train/eval JSONLs are archived (or at minimum content-addressed by hash
   into `datasets/built/<sha12>/`) so "regenerated the dataset mid-run" can't destroy
   evidence.
2. **The benchmark is a fixed, versioned artifact.** Reservoir-sampled, seeded,
   state-mapped from day 1 (already fixed on master — carry over). The cache gets a
   schema version + content hash; a result is only comparable to results on the same
   cache hash, and the report says so.
3. **No train/benchmark overlap by construction.** Today the 4 ID sources are
   re-sampled from the same files training reads (TASK-015: they measure retention,
   not generalization). Redo: split at the *capture/file* level — hold out whole
   files/scenarios per source for the ID benchmark. Row-level dedup is unreliable
   (stochastic masking); file-level holdout is clean and cheap.
4. **Select checkpoints by the deployment metric.** eval_loss mis-selects (V11 ep2).
   During training, compute verdict-level metrics (MCC on the eval split via
   constrained verdict decoding or first-token logit comparison) and use that for
   `metric_for_best_model`. OOD probes stay out of selection (they're the test set),
   but ID-eval MCC ≥ eval_loss as a selector.
5. **Composition is declarative.** One table in one config: per-source attack/benign
   quota as a *fraction of the final budget* (e.g. `ctu_malware: 0.20 of attacks`),
   resolved once, printed, and asserted (`assert every source ≤ its quota + ε`).
   Loaders just yield rows; no per-file caps interacting with global weights, no
   TRAINING_FACTOR rescaling caps (the exact v11 accident mechanism is removed —
   volume scaling is done only by downsampling the *composed* dataset, per
   preprocess_downsample.py's rationale). SF-oversampling, if kept, is expressed as
   an explicit quota on a (source, state-class) cell, not a hidden 2× draw weight.
6. **Statistics are part of the report.** Wilson 95% CIs on every recall, McNemar
   p-values on paired model deltas, prevalence-adjusted precision @ 1:100 and 1:1000
   priors, and separate `mcc_id` / `mcc_ood` — i.e. TASK-013/014/015 are the default
   report, not open tasks.
7. **One prompt builder, one parser, one state map** — this already exists
   (`prompt_utils`, `zeek_log_utils`) and is the part of the old codebase that worked;
   carry it over rather than rewriting for its own sake.
8. **Every loader ships a golden-file test.** The UWF port bug (silent N/A for three
   versions) and the IoT-23 field-count off-by-one both die to a 20-line fixture test
   per source: tiny checked-in sample file → exact expected row dicts. Add a
   composition smoke test (build at tiny scale, assert quotas/ratio/reason-format).

---

## 2. Method redesign

### 2.1 Grounded REASON (the headline change)

Replace `pick_reason()` (random from 77/48 pools) with a **deterministic evidence
templater**: a small rule engine that reads the *actual* flow features and emits the
reason citing them, e.g.

- `S0 + resp_bytes=0 + dur≈0` → "SYN probe with no response — consistent with a port
  scan against a closed or filtered port."
- `REJ/RSTR + orig_bytes_per_pkt=0 + AD port (88/135/389/445/3389/464)` → "refused
  connection to a Windows domain service port — lateral-movement probing pattern."
- benign `UDP S0, 700–1500B, dest port ≥49152` → "link-local multicast name
  resolution broadcast (LLMNR/mDNS/NetBIOS) with no responder."

Rules are derived from the existing pools (they already encode the domain knowledge —
the change is selecting by evidence instead of at random) plus the hard-benign flags
(`score_hard_benign` already computes exactly these evidence flags — reuse them as the
reason selector). A generic fallback covers unmatched flows. Every reason must be
*true of the sample it's attached to* — that is the property the old design lacked.

**Ordering ablation:** train arms with (a) `VERDICT → REASON` (status quo order),
(b) `REASON → VERDICT` (CoT-style: the reason can now causally help the verdict;
parse verdict from last line), (c) verdict-only (cheapest; control). Compare on MCC.
(b) is the scientifically interesting arm; if it wins it's a new thesis contribution
("evidence-grounded rationales improve flow classification"), and if it doesn't, the
grounded-reason model still fixes "confidently wrong" for the deployment story.

**Evaluation of reasons (thesis Q3):** with grounded reasons, reason *faithfulness*
becomes checkable automatically — re-run the rule engine on the input and test whether
the model's emitted reason matches the evidence class. Report faithfulness % ID vs
OOD. This turns the "confidently wrong" finding from anecdote into a measured metric.

### 2.2 Everything else held or simplified

- Base model, QLoRA 4-bit NF4, 7 target modules: unchanged. LoRA r: **decided by
  v13** (r=16 vs r=32 A/B is running under the old plan — its answer transfers).
- Packing: **decided by v13.1** (the SDPA cross-sample-attention concern). If no-pack
  wins or ties, default no-pack; if pack wins, either accept or add flash-attn on the
  RunPod path only.
- Epochs: default 1 with step-based checkpointing + ID-MCC selection (v12.2's loss
  curve shows saturation inside epoch 1; V11 showed ep2 hurts OOD). 2-epoch arm only
  if v13.2 contradicts this.
- max_length 512, masking probs (0.20 state / 0.50 context), 2:1 ratio: carry over
  unchanged — all evidence-backed.
- [BEHAVIOR]/multi-log context: keep the *capability* (it's the deployment story and
  costs little) but document as null-result for OOD; no new investment.
- Host Pass-2: drop from the new codebase entirely (documented null result; the old
  repo preserves it for reproducibility).

### 2.3 Data / probes (allowed to change, minimal changes proposed)

- **Keep** the 6 training sources; they're fine. Consider dropping UNSW-NB15 in an
  ablation only if a cheap run shows it's neutral (it's the most synthetic source).
- **Keep** Win7AD-1 (hard) and Kelihos (floor, honest ~40-44%).
- **Echo:** its ground truth is questionable (honeypot inbound scans auto-labelled
  Malicious; the "RETIRED" comment vs active use contradiction in TASK-016). Redo:
  keep it in the report but *out of the headline OOD MCC*, or replace it with a
  second clean probe (another CTU-SME-11 device capture with outbound-infection
  ground truth, like Win7AD-1). Decide during implementation after a label audit of
  ~50 Echo samples.
- ID benchmark drawn from **held-out files** per source (principle 3).

---

## 3. New codebase shape — a CLEAN, SEPARATE REPO

The redo lives in a **new repository** that recreates this project from scratch,
targeting better results. This repo (`fine_tunning`) stays untouched as the frozen
reference: old models, old benchmark numbers, thesis notes, git history.

The new repo must be self-bootstrapping — everything regenerable from public sources:

```
<new-repo>/
├── README.md            # bootstrap: download → build → train → benchmark, start to finish
├── config.py            # ONE declarative config: composition table, masking, paths, seeds
├── download.py          # fetch all raw datasets + OOD captures (port of download_datasets.py,
│                        #   extended to cover test_captures/ Zenodo + Botnet-3 downloads too)
├── sources/             # one loader per source, all yielding the same Row dataclass
├── build.py             # compose dataset: quotas → split (file-level holdout) → write + manifest
├── reasons.py           # grounded-reason rule engine (+ faithfulness checker)
├── prompt.py            # build_prompt / system prompts / parsers  (ported)
├── train.py             # SFTTrainer wrapper; manifest-before-train; ID-MCC selection
├── bench/
│   ├── loaders.py       # reservoir bench loaders (ported, held-out-file aware)
│   ├── run.py           # inference + report (CIs, McNemar, prevalence precision, mcc_id/ood)
│   └── baseline_ml.py   # RF/LR on the same cache (ported)
├── deploy/              # merge_adapter + Modelfile + Ollama path (ported, trimmed)
├── provenance.py        # manifests, hashes, ledger (recursive scan — TASK-017 fixed by design)
└── tests/               # golden-file test per loader + composition smoke + reason-engine tests
```

Porting policy: **port, don't rewrite, the proven parts** (prompt_utils,
zeek_log_utils parsing/state map, behavior_features, reservoir sampler, infer_utils,
download_datasets) — copied in with their lessons intact, then owned by the new repo.
Everything else (preprocess pipeline, train wrapper, benchmark orchestration,
provenance) is written fresh to the principles in §1.

Practical bridges to the old repo (no code dependency, just data reuse):
- `datasets/` and `test_captures/` can be symlinked or copied locally to skip
  re-downloading ~11 GB; `download.py` remains the canonical path for a cold machine.
- Cross-repo comparability: the new benchmark reports its cache hash; to compare a
  new model against V11 ep1, run the old checkpoint through the *new* benchmark once
  (adapters are just directories — loadable from any path). Old-repo numbers are
  never mixed into new-repo tables without that re-run.

---

## 4. Experiment ladder (cheap → expensive, each gated)

Costs assume ~$2–3/full RunPod run or ~1 day local; total ladder ≈ 4–6 full runs.

| # | Run | Question | Gate to next |
|---|-----|----------|--------------|
| 0 | (already running) v12.2 / v13 / v13.1 / v13.2 under the *old* code | data volume; r=16 vs 32; pack vs no-pack; 1 vs 2 epochs | **Revise this plan with the answers** — they fix the redo's default r / packing / epochs / dataset size for free |
| 1 | Rebuild dataset with `ids2` (grounded reasons, declarative composition, file-level holdout) at the volume v12.2 blessed; verify composition + goldens | pipeline correctness | composition asserts pass; token stats ≈ old |
| 2 | **R1**: train verdict→reason (grounded), defaults from #0 | does grounding alone hold/beat V11 ep1 MCC +0.777 on the fixed benchmark? | ≥ V11 ep1 within CI |
| 3 | **R2**: reason→verdict (CoT order) | does evidence-first reasoning improve MCC? | compare by MCC + McNemar |
| 4 | **R0**: verdict-only control | quantifies reason cost/benefit cleanly (the never-run A/B) | — |
| 5 | Reason-faithfulness eval on R1/R2, ID vs OOD | turns "confidently wrong" into a measured metric | — |
| 6 | (optional, last) Qwen2.5-3B control at winning config | "capacity is not the bottleneck" confirmation for the thesis | only if time/budget allow |

Ship = best-by-ID-MCC checkpoint of the winning arm, then OOD-benchmarked, merged →
GGUF → Ollama with the matching SYSTEM prompt (auto-resolved from run.json as today).

## 5. Thesis-first deliverables (independent of training)

- Re-baselined, CI-annotated results tables generated straight from the new report
  code (feeds thesis draft_1.md §5–6 tables).
- Explicit ID-(retention) vs OOD-(generalization) framing everywhere (TASK-015).
- Honest-number corrections propagated: Kelihos ~40-44%, Win7AD-1 83% (fixed bench),
  the V10 87.1% flagged as measured under the old buggy sampling (open item from
  thesis_notes_13 — optionally re-benchmark the V10 checkpoint, it still exists in
  `models/v10-ids-model`, one cheap inference pass).
- Grounded-reason + faithfulness metric section replaces "REASON is random" as the
  method; "confidently wrong" stays as the motivating finding.

## 6. Open until v13 lands (placeholders to edit)

- [ ] LoRA r default: `r=__` (from v13 vs v11 comparison)
- [ ] Packing default: `__` (from v13 vs v13.1)
- [ ] Epochs default: `__` (from v13.2)
- [ ] Dataset volume: `full | 50%` (from v12.2 vs v12.1)
- [ ] Echo probe: keep / demote / replace (after label audit)
- [ ] UNSW keep/drop ablation: run only if a slot is free
