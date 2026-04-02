# Review Tasks

## Context
- Review branch: master
- Implementation branch: development
- Review date: 2026-04-02

---

## Open Tasks

### TASK-006
Status: open
Priority: high
Area: train.py — lines 40–41 and 113
Problem: `OUTPUT_DIR` and `ADAPTER_DIR` still point to v9.0 paths. Running training
now would write checkpoints to `./v9.0-ids-model` and the final adapter to
`./v9.0-ids-lora-adapter`, risking an accidental overwrite of the existing v9.0 adapter.
Required fix: Make exactly these three line replacements in train.py:
  Line 40: `OUTPUT_DIR   = "./v9.0-ids-model"`
        → `OUTPUT_DIR   = "./v9.1-ids-model"`
  Line 41: `ADAPTER_DIR  = "./v9.0-ids-lora-adapter"`
        → `ADAPTER_DIR  = "./v9.1-ids-lora-adapter"`
  Line 113 comment: `# v9.0: extended for multi-log prompts (http/dns/ssl context)`
        → `# v9.1: extended for multi-log prompts (http/dns/ssl/behavior context)`
Validation: `grep -n "v9\." train.py` should show only v9.1 references after the fix.
`python -m py_compile train.py` must pass.
Notes: Do not change any other hyperparameters, config, or logic.

---

### TASK-007
Status: open
Priority: low
Area: REVIEW_TASKS.md
Problem: TASK-001 through TASK-005 are listed as open but were all completed in
commit 504407c. The file needs to reflect the actual state.
Required fix: Replace the current Open Tasks section (TASK-001 through TASK-005)
with just TASK-006 and TASK-007, and move TASK-001 through TASK-005 into the
Completed Tasks section with Status changed from "open" to "done" on each.
No other edits to the task text.
Validation: Only TASK-006 and TASK-007 appear under Open Tasks. TASK-001 through
TASK-005 appear under Completed Tasks with Status: done.
Notes: Do TASK-006 first, then update this file last so the final open-tasks list
is accurate.

---

### TASK-008
Status: open
Priority: critical
Area: loader_ctu_malware.py — lines 80–124
Problem: `conn_rows` is never hard-capped. The two-pass design buffers all rows in
pass 1, then processes them in pass 2. The early-exit condition in pass 1 only breaks
when BOTH `buffered_counts["ATTACK"] >= attack_cap` AND
`buffered_counts["FALSE POSITIVE"] >= benign_cap` are satisfied simultaneously.
For attack-heavy CTU-Malware scenarios (e.g. Botnet-42 Ramnit, Botnet-90 Pushdo)
the conn.log may have tens of thousands of "Botnet" flows but only hundreds of
"Normal" flows — `benign_cap` is never reached and the loop reads the entire
conn.log (potentially hundreds of MB, millions of rows) into memory before breaking.
`build_behavior_contexts(conn_rows)` is then called on the full set, compounding the
time. This is the observed freeze / apparent infinite loop.
The same structural pattern exists in `loader_iot23.py` (lines 92–97) and
`loader_unsw.py` (lines 101–106), though those sources are better balanced so the
condition is usually satisfied early.
Required fix: Hard-cap `conn_rows` at `row_cap` unconditionally — break immediately
when `len(conn_rows) >= row_cap` without any condition check. Remove the conditional
check inside that block entirely. The second-pass filtering via `MAX_PER_SOURCE_CLASS`
already enforces the correct per-bucket limit, so the only purpose of the first-pass
early exit is to avoid reading huge files; a hard cap on rows is sufficient.
Apply the same hard-cap fix to `loader_iot23.py` and `loader_unsw.py` for consistency.
Validation: With `TRAINING_FACTOR=0.03` (row_cap ≈ 19,200 for CTU-Malware), the
CTU-Malware loader must complete all 7 scenarios in under 60 seconds on an SSD.
No scenario should cause `len(conn_rows)` to exceed `row_cap`.

---

### TASK-009
Status: open
Priority: high
Area: zeek_log_utils.py — line 69
Problem: `urllib.request.urlretrieve(url, local)` does not support a transfer timeout.
The `urlopen(..., timeout=30)` in `find_binetflow_url` covers the initial connect only.
If the Stratosphere Lab server stalls mid-transfer (common for large conn.log files,
some of which are 100–500 MB), the download hangs indefinitely with no way to
interrupt it short of SIGKILL. This is a secondary cause of the CTU-Malware freezes.
Required fix: Replace `urlretrieve` with a manual streaming download using
`urllib.request.urlopen(url, timeout=60)` and writing chunks in a loop. This allows
the 60 s timeout to apply to each socket read, not just the initial connection.
Validation: Temporarily point `ctu_download` at an unreachable or throttled URL;
the call must raise `socket.timeout` (or similar) within ~60 s rather than hanging.

---

### TASK-010
Status: open
Priority: medium
Area: preprocess_zeek.py — lines 65–76
Problem: All six loaders run sequentially in a single process, using one CPU core.
Total wall-clock time on an 8-core Ryzen 7 3700X at full dataset is dominated by
this sequential bottleneck: IoT-23 (tar.gz), CTU-13 (tar.bz2), UNSW (parquet I/O),
UWF (CSV), CTU-Normal (TSV), and CTU-Malware (download + parse) are all independent
with no shared state. Each also calls `build_behavior_contexts`, which is pure Python
and CPU-bound.
Required fix: Wrap each loader call in `concurrent.futures.ProcessPoolExecutor` with
`max_workers = min(6, os.cpu_count())`. Each loader must be importable as a
standalone callable (they already are — each loader function takes only its path
argument). Collect results with `executor.map` or `as_completed` and extend
`all_samples` after all futures resolve. Note: `random.seed(RANDOM_SEED)` must be
re-applied in each worker process since forked processes do not inherit PRNG state
reliably across all platforms.
Validation: `time .venv/bin/python preprocess_zeek.py` wall-clock time must decrease
by at least 2× on a 4+ core machine compared to sequential baseline.

---

## Completed Tasks
