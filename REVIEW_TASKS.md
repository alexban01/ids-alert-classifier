# Review Tasks

## Context
- Review branch: master
- Implementation branch: development
- Review date: 2026-07-02
- Scope: methodology review of `benchmarks/benchmark_realworld.py` + `benchmarks/bench_loaders.py`
- TASK-011 and TASK-012 landed directly on master 2026-07-02 (one-off exception to
  the usual Codex-on-development workflow, at the user's request, to unblock a
  clean v12 benchmark same-day). Both invalidated the sample cache — do ONE
  `--regen` before trusting any V11 vs V12 comparison.

---

## Open Tasks

### TASK-013
Status: open
Priority: medium
Area: benchmarks/benchmark_realworld.py — compute_metrics (line 334),
print_comparison_table (line 413)
Problem: No uncertainty quantification. At n=300 per class a recall estimate
carries roughly ±3–5 pp at 95% confidence, so per-source deltas of a few points
between checkpoints are within noise, and the report presents them as if exact.
Required fix: Add Wilson 95% confidence intervals to atk_recall and ben_recall
(overall and per-source) — implementable in a few lines of stdlib `math`, no new
dependency. Show as `88.0% [84.1–91.1]` in the per-source table and comparison
table. Additionally, in `print_comparison_table`, run McNemar's test on paired
predictions for each adjacent model pair (preds are already kept in
`m["preds"]`/`m["truths"]`) and print the p-value under each delta row.
Validation: Report shows CI brackets on all recall figures; comparison deltas
carry a McNemar p-value; `results/benchmark_realworld_results.json` includes
`atk_recall_ci` / `ben_recall_ci` per model and per source. Metrics values
themselves are unchanged from before.
Notes: scipy is available (sklearn dependency) if preferred for McNemar, but the
exact binomial form is ~5 lines of stdlib.

---

### TASK-014
Status: open
Priority: medium
Area: benchmarks/benchmark_realworld.py — print_comparison_table (line 413),
json_output model records (line 627)
Problem: The benchmark's 50/50 class balance doesn't reflect deployment, where
alert streams are overwhelmingly benign. MCC and precision on a balanced sample
are optimistic: at a 1:100 benign:attack prior, 95–98% benign recall still buries
analysts in false alerts. The report makes no mention of this.
Required fix: Add a prevalence-adjusted precision line per model, computed
analytically from the measured recalls (no re-run needed):
`precision(p) = (atk_recall·p) / (atk_recall·p + (1−ben_recall)·(1−p))`
for p ∈ {0.01, 0.001}. Print in the comparison-table footer (labelled clearly,
e.g. "Est. precision @ 1:100 prior / @ 1:1000 prior") and store both values in
each model's record in `results/benchmark_realworld_results.json`.
Validation: Footer shows both figures per model; hand-check one against the
formula using the reported recalls. Existing metrics unchanged.
Notes: This is a reporting addition only — do not change sampling balance.

---

### TASK-015
Status: open
Priority: medium
Area: benchmarks/benchmark_realworld.py — report header (line 531) and comparison
footer (line 444)
Problem: The four in-distribution sources (iot23, ctu13, uwf, ctu_normal) are
re-sampled from the same raw files the training loaders read, and both read from
the head of those files — so benchmark rows very likely overlap the training pool.
Those sources measure retention ("did training stick"), not generalization; only
the three OOD probes measure generalization. The report doesn't distinguish, and
a reader will read the overall MCC as a generalization number.
Required fix: Label the two groups explicitly in the report: header and footer
should describe iot23/ctu13/uwf/ctu_normal as "in-distribution (retention —
sources overlap the training pool)" and win7ad/sme11/botnet3 as "OOD
(generalization — never in training)". Additionally print a two-line summary
after the comparison table: overall MCC recomputed separately over ID-only and
OOD-only subsets per model (data is already in `per_source`/preds — no extra
inference), and store `mcc_id` / `mcc_ood` in the results JSON.
Validation: Report clearly separates the two groups; `mcc_ood` for a model
matches the MCC from a `--ood` run of the same model on the same cache.
Notes: Documentation + derived metrics only. TASK-012's random sampling reduces
(but does not eliminate) row-level overlap; do not attempt row-level dedup against
`zeek_dataset.jsonl` — training prompts have stochastic masking applied, so exact
matching would be unreliable. Framing honestly is the fix.

---

### TASK-016
Status: open
Priority: low
Area: benchmarks/bench_loaders.py line 42; benchmarks/benchmark_realworld.py line 454
Problem: Two stale/inaccurate comment strings. (1) `bench_loaders.py:42` says
"Echo (RETIRED): inbound internet background scanning, invalid OOD ground truth"
— but Echo is active in `ALL_SOURCES`, counted in the headline MCC, and treated
as the valid OOD-Easy probe by STATE.md and the loader's own docstring. (2) The
report footer claims "NO synthetic field mapping", but CTU-13 splits `TotPkts`
50/50 into orig/resp packet counts and derives resp_bytes — a synthetic mapping.
Required fix: (1) Rewrite the line-42 comment to match reality: Echo = easy-OOD
probe (inbound IoT scan traffic), active. (2) Soften the footer to
"(native Zeek conn.log; CTU-13 binetflow mapped — packet counts split 50/50)".
No logic changes.
Validation: `grep -n "RETIRED" benchmarks/` returns nothing;
`grep -n "NO synthetic" benchmarks/` returns nothing. `python -m py_compile`
passes on both files.
Notes: If Echo's ground truth genuinely is questionable (honeypot inbound scans
auto-labelled Malicious), that's a thesis decision to exclude it from headline
MCC — flag to the user rather than deciding here; this task only fixes the
contradiction between the comment and current behaviour.

---

### TASK-017
Status: open
Priority: medium
Area: ids/run_manifest.py — regenerate_experiments_md, line 234
Problem: `regenerate_experiments_md()` finds runs via `os.listdir(MODELS_DIR)` then
reads `models/<name>/run.json` — one level deep only. It silently misses `run.json`
nested under a checkpoint dir (`models/<name>/checkpoint-N/run.json`), which is
where it lives for any adapter whose training run never finished into a top-level
`<name>-ids-lora-adapter/` dir. Confirmed 2026-07-02: v12 and v12.1 (both
interrupted mid-training, checkpoint-only) have real `benchmark.mcc` values in
their `run.json`, but `EXPERIMENTS.md` printed "(no runs yet)" after a rebuild —
`attach_benchmark_result()` writes to the correct nested path and even calls
`regenerate_experiments_md()` right after, which then finds 0 rows and silently
overwrites the ledger with an empty table. No error, no warning.
Required fix: Have `regenerate_experiments_md()` also recurse one level into each
`models/<name>/` subdirectory looking for `checkpoint-*/run.json`, in addition to
the existing `models/<name>/run.json` check. Both a top-level and nested run.json
should never coexist for the same adapter in practice, but if they do, prefer the
top-level one (it reflects a completed run).
Validation: With `models/v12-ids-model/checkpoint-10000/run.json` and
`models/v12.1-ids-model/checkpoint-10000/run.json` present (both have
`benchmark.mcc`), `.venv/bin/python scripts/experiments.py` produces an
`EXPERIMENTS.md` with rows for both, not "(no runs yet)". Existing top-level-only
adapters (if any exist at rebuild time) still appear unchanged.
Notes: Low urgency — MCC is still readable directly from the checkpoint's
`run.json` or the benchmark report in the meantime. See STATE.md "Known gap" note
under Experiment tracking & provenance.

---

### TASK-018
Status: open
Priority: high (blocks v13.3 training run)
Area: train.py — model load in the training path; requirements/setup notes
Problem: TRL 0.29.1's default packing (`packing_strategy="bfd"`) only isolates
packed samples from each other when the model runs a flash-attention variant
that respects sequence boundaries. This repo never sets `attn_implementation`
(model runs SDPA/eager) and has no flash-attn installed, so packed samples
(~1.46 per 512-token slot) attend across sample boundaries during training.
The v12.2 (packed, MCC +0.7308) vs v13.1 (no-pack, +0.7465) A/B is consistent
with this leak hurting quality (thesis_notes_14.txt §2), but packing is ~3.4×
faster per epoch — worth recovering if it can be made correct.
Required fix: Add a `--flash-attn` flag to train.py that (1) passes
`attn_implementation="flash_attention_2"` to the base-model `from_pretrained`
call, and (2) records `"attn_implementation"` in the run.json hyperparams dict
(record the resolved value on every run, including the SDPA default, so runs
are distinguishable in the ledger). Error out early with a clear message if
`--flash-attn` is passed but `flash_attn` is not importable. Do not change the
default (SDPA) behaviour when the flag is absent.
Environment prerequisite (owner runs this, not part of the code change):
`.venv/bin/pip install flash-attn --no-build-isolation` — needs nvcc matching
the venv's torch CUDA version; source build can take 30+ min, set MAX_JOBS to
bound RAM. RTX 3070 is Ampere (SM 8.6), supported by FlashAttention-2; compute
dtype is already bf16, which FA2 requires.
Validation: `train.py --flash-attn --dataset zeek_dataset_50pct.jsonl
--save-steps 50` starts and completes ≥50 steps with packing on, no attention-
mask warnings from transformers, and the saved checkpoint's run.json contains
`"attn_implementation": "flash_attention_2"`. A default run (no flag) is
byte-identical in behaviour to today and records `"attn_implementation": "sdpa"`.
Notes: Inference/benchmark paths need no change — the adapter is just weights;
`ids/infer_utils.py` can keep loading with default attention. Purpose is the
v13.3 experiment (STATE.md Next steps): rerun the v12.2 config (r=16, packed,
1 epoch, 50% data) with correct packing to isolate whether the attention leak
caused packing's quality loss.
