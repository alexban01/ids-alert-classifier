# Review Tasks

## Context
- Review branch: master
- Implementation branch: v9.1
- Review date: 2026-04-01

---

## Open Tasks

### TASK-001
Status: open
Priority: high
Area: preprocess_zeek.py — `make_sample()`
Problem: `behavior_ctx` is always passed to `build_prompt()` when available. HTTP, DNS,
and SSL context are randomly dropped at `CONTEXT_MASK_PROB = 0.50` to prevent the model
from learning "has context section → ATTACK". `[BEHAVIOR]` has no equivalent masking,
so the model can learn "has BEHAVIOR section → richer prompt → probably ATTACK" instead
of learning to use the behavioral signals themselves.
Required fix: In `make_sample()`, apply the same `CONTEXT_MASK_PROB` dropout to
`behavior_ctx` before passing it to `build_prompt()`, the same way `http_ctx`,
`dns_ctx`, and `ssl_ctx` are masked. Keep the raw `behavior_ctx` in the returned
sample dict (for `hard_benign_score`), only mask the prompt argument.
Validation: After the fix, check that the context-coverage diagnostic printed at the
end of `preprocess_zeek.py` shows `[BEHAVIOR]` appearing in ~50% of samples for both
attacks and benign, similar to the other context sections.
Notes: `CONTEXT_MASK_PROB` is already defined at module level — reuse it, do not
introduce a separate constant.

---

### TASK-002
Status: open
Priority: high
Area: preprocess_zeek.py — `load_iot23()`, `load_ctu_normal()`, `load_ctu_malware_captures()`
Problem: All three loaders now buffer every conn.log row into a list before calling
`build_behavior_contexts()`. For large IoT-23 members (some have millions of rows),
this replaces the previous streaming path with an unbounded in-memory list.
A large member can exhaust the 32 GB RAM before any cap is applied.
Required fix: Add a row-collection cap inside each loader's inner loop. Stop appending
to `rows` once the relevant per-verdict buckets are already at their caps AND a
sufficient buffer for behavior context is held. A safe heuristic: stop collecting rows
once `len(rows) >= (attack_cap + benign_cap) * 4`. Apply this cap check at the
point where rows are appended to the `rows` list, before `build_behavior_contexts()`
is called.
Validation: Run `preprocess_zeek.py` on the full dataset and confirm peak RSS stays
within the previous range. If the full dataset isn't available, confirm the cap
triggers correctly on a small synthetic test (a list of 500k dummy rows where the
cap limit is 200k).
Notes: The cap multiplier of 4 gives enough overflow rows for the behavior window to
have meaningful history without unbounded allocation. The exact value can be tuned;
the goal is a hard upper bound per loader invocation.

---

### TASK-003
Status: open
Priority: medium
Area: benchmark_realworld.py — `rebuild_prompts_with_behavior()`, main block
Problem: `rebuild_prompts_with_behavior()` is called unconditionally after
`generate_samples()`. There is no way to run the benchmark without behavior context
to establish a baseline comparison. The v9.1 plan explicitly requires ablation
comparisons (conn-only vs. conn+behavior) to validate whether the behavior features
actually help.
Required fix: Add a `--no-behavior` CLI flag. When present, skip the call to
`rebuild_prompts_with_behavior()` so prompts use only base conn.log fields (matching
the v8/v9.0 baseline). The flag should be detected alongside the existing `--regen`
flag. No other logic changes needed.
Validation: Run `benchmark_realworld.py --no-behavior` and confirm the printed prompts
contain no `[BEHAVIOR]` section. Run without the flag and confirm `[BEHAVIOR]` appears
for sources that provide timestamps (IoT-23, CTU-Normal, CTU-Botnet3, UWF).
Notes: This is the minimum needed to do a before/after comparison without switching
branches. The flag name must be `--no-behavior` to stay consistent with existing
`--regen` naming style.

---

### TASK-004
Status: open
Priority: medium
Area: preprocess_zeek.py — `load_unsw()`, `load_uwf()`
Problem: Three of the five training sources (UNSW-NB15, UWF-ZeekData24, CTU-13) have
no behavior context attached. CTU-Malware and IoT-23 attacks get `[BEHAVIOR]` sections;
UNSW attacks and UWF benigns do not. This creates a labeling asymmetry: if attacks
disproportionately carry behavior context, the model may learn context-presence as a
proxy for ATTACK verdict, undermining the masking in TASK-001.
Required fix: Apply `build_behavior_contexts()` in `load_unsw()` and `load_uwf()` the
same way it was applied in `load_iot23()` and `load_ctu_normal()`. CTU-13 (binetflow)
does not have reliable per-flow timestamps, so skip it.
For `load_unsw()`: rows already have a `ts` column in the parquet — use it as the
timestamp. Collect rows into a list, call `build_behavior_contexts(rows)`, then
iterate `zip(rows, behavior_ctxs)` to call `make_sample()` with `behavior_ctx`.
For `load_uwf()`: same pattern; the CSV has a `ts` field.
Validation: After the change, the context-coverage diagnostic at the end of
`preprocess_zeek.py` should show `[BEHAVIOR]` present in both attack and benign
train pools at comparable rates across sources.
Notes: Apply the same row-collection cap from TASK-002 to avoid memory issues. Do
TASK-002 first, then TASK-004 can reuse the same cap pattern.

---

### TASK-005
Status: open
Priority: low
Area: behavior_features.py — `build_behavior_contexts()`
Problem: The sliding-window temporal logic has several edge cases with no test coverage:
rows without timestamps, single-row input, rows that arrive out of chronological order
across different sources, and the periodicity label transitions (Low/Medium/High).
A silent bug here corrupts training data and inference results.
Required fix: Add a `test_behavior_features.py` file with pytest-style tests covering:
1. Empty input returns `[]`.
2. Single row returns `[context]` where all window counts are 0.
3. Two rows from the same source 30 seconds apart: second row sees `src_conn_60s == 1`.
4. Rows with `ts = None` or `ts = "-"` produce `None` context entries (not a crash).
5. `_periodic_label` with 3 identical gaps (e.g., 30.0, 30.0, 30.0) returns `"High"`.
6. `_periodic_label` with fewer than 3 gaps returns `"Low"`.
Validation: `python -m pytest test_behavior_features.py -v` passes with no failures.
Notes: Use only stdlib — no numpy or pandas needed. Keep the test file in the project
root alongside the other scripts.

---

## Completed Tasks
<!-- move finished tasks here instead of deleting them immediately -->
