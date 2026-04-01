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

## Completed Tasks
