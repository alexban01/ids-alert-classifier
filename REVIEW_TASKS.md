# Review Tasks

## Context
- Review branch: master
- Implementation branch: development
- Review date: 2026-04-02

---

## Open Tasks

### TASK-008
Status: open
Priority: high
Area: preprocess_zeek.py — load_unsw() line 1116, load_uwf() line 1298
Problem: Both loaders iterate pandas DataFrames with `df.iterrows()`, which is the
slowest pandas iteration method — it constructs a full Series per row with type
inference overhead. At full scale (UNSW contributes up to 80k rows, UWF up to 80k),
this is a significant bottleneck.
Required fix: Replace `for _, row in df.iterrows():` with `for row in df.itertuples(index=False):`
in both `load_unsw()` and `load_uwf()`. Column access changes from `row[col]` (string key)
to `getattr(row, col)` or `row._asdict()[col]`. Because column names may contain dots or
spaces (UWF has `id.orig_h`, `id.resp_h`, `id.orig_p`, `id.resp_p`), use
`row._asdict().get(col, default)` for any column access that might have special characters.
No other logic changes.
Validation:
  python -m py_compile preprocess_zeek.py
  TRAINING_FACTOR=0.03 .venv/bin/python preprocess_zeek.py
  Verify UNSW and UWF sample counts match the pre-change run.
Notes: Do not change load_cicids() — it already uses iterrows() but CICIDS is
currently disabled, so it is not a priority.

---

### TASK-009
Status: open
Priority: medium
Area: preprocess_zeek.py — load_ctu13(), DATASETS dict
Problem: CTU-13 is loaded from a `.tar.bz2` archive using Python's single-threaded
bz2 decompressor, which is the slowest loader at full scale. The archive can be
pre-extracted once with an external tool (pbzip2, lbzip2, or plain tar) to eliminate
the decompression cost on every run. The loader needs to support both the archive
and a pre-extracted directory.
Required fix: At the top of `load_ctu13(archive_path)`, before opening the archive,
check whether a pre-extracted directory exists alongside the archive:
  extracted_dir = archive_path.replace(".tar.bz2", "")
  # e.g. "datasets/ctu-13/CTU-13-Dataset" if archive is "datasets/ctu-13/CTU-13-Dataset.tar.bz2"
If `os.path.isdir(extracted_dir)` is True, collect `.binetflow` files from it with:
  members_paths = sorted(glob.glob(os.path.join(extracted_dir, "**/*.binetflow"), recursive=True))
Then iterate over those plain file paths using `open(path)` instead of
`tf.extractfile(member)`. If the directory does not exist, fall back to the existing
`tarfile.open(archive_path, "r:bz2")` code unchanged.
Print `[CTU-13] Using pre-extracted directory: {extracted_dir}` or
`[CTU-13] Decompressing archive: {archive_path}` accordingly.
Validation:
  python -m py_compile preprocess_zeek.py
  # With archive only (existing behavior unchanged):
  TRAINING_FACTOR=0.03 .venv/bin/python preprocess_zeek.py
  # After manually extracting:
  # cd datasets/ctu-13 && tar -xjf CTU-13-Dataset.tar.bz2
  # Re-run and verify same sample counts, faster wall time.
Notes: Do not modify the DATASETS dict or any other loader. The directory detection
is purely inside load_ctu13().

---

### TASK-010
Status: open
Priority: medium
Area: preprocess_zeek.py — main block (~line 1474), all loader functions
Problem: The six loaders (load_iot23, load_ctu13, load_unsw, load_uwf,
load_ctu_normal, load_ctu_malware_captures) run sequentially in main. They are
fully independent and can run in parallel. On an 8-core Ryzen 3700X, overlapping
I/O-heavy decompression and pandas loading would cut wall time roughly in half.
Required fix:
1. Add a `seed` parameter (int) to each of the six loader functions. At the very
   start of each loader body, call `random.seed(seed)`. This ensures each worker
   process has deterministic but distinct random state for pick_reason() and
   make_sample() masking. Do not change any other logic inside the loaders.
2. In main, replace the sequential loader calls with a
   `concurrent.futures.ProcessPoolExecutor(max_workers=4)` block:
     from concurrent.futures import ProcessPoolExecutor
     loader_args = [
         (load_iot23,               (DATASETS["iot23"],),        RANDOM_SEED + 1),
         (load_ctu13,               (DATASETS["ctu13"],),        RANDOM_SEED + 2),
         (load_unsw,                (DATASETS["unsw"],),         RANDOM_SEED + 3),
         (load_uwf,                 (DATASETS["uwf"],),          RANDOM_SEED + 4),
         (load_ctu_normal,          (DATASETS["ctu_normal"],),   RANDOM_SEED + 5),
         (load_ctu_malware_captures, (),                         RANDOM_SEED + 6),
     ]
     def _run_loader(fn, args, seed):
         return fn(*args, seed=seed)
     with ProcessPoolExecutor(max_workers=4) as executor:
         futures = [executor.submit(_run_loader, fn, args, seed)
                    for fn, args, seed in loader_args]
         all_samples = []
         for f in futures:
             all_samples += f.result()
3. The module-level `random.seed(RANDOM_SEED)` call at line 315 and the one in
   main (line 1476) must remain so the post-load shuffle and sampling in main
   stays deterministic on the main process.
4. Wrap the main block contents in `if __name__ == "__main__":` — this is required
   for ProcessPoolExecutor on Linux with the fork start method to avoid re-executing
   top-level code in worker processes. If it is already wrapped, leave it as-is.
Validation:
  python -m py_compile preprocess_zeek.py
  TRAINING_FACTOR=0.03 .venv/bin/python preprocess_zeek.py
  Verify total sample counts match the sequential run (within ±5% — seeding
  differences from parallelism may cause minor variation in random masking).
  Verify no deadlocks or import errors on startup.
Notes: Do not use multiprocessing.Pool or threading — ProcessPoolExecutor is
preferred for GIL-bound CPU work. max_workers=4 leaves cores free for the OS
and avoids memory contention when all loaders run at once on 32 GB RAM.
If a loader raises an exception, f.result() will re-raise it in main — no silent
failures. Do not catch exceptions inside _run_loader.

---
