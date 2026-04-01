# Agent Collaboration Contract

This repository uses two coding agents with distinct responsibilities:

- **Codex / GPT** = implementation agent
- **Claude / Sonnet** = review agent

These rules are part of the project contract and apply unless the user explicitly overrides them in the current session.

## Active Branch Roles

- **Review branch:** `master`
- **Implementation branch:** `development`

When the implementation branch changes in the future, update only this section.

---

## Global Rules (applies to both agents)

1. Do not make up branch names, tasks, or requirements.
2. Do not modify both branches interchangeably. Respect branch ownership.
3. Prefer small, reviewable changes over large rewrites.
4. Do not touch unrelated files.
5. Preserve public behavior unless the task explicitly requires a behavior change.
6. Run the smallest relevant validation for the change.
7. If a rule here conflicts with a direct user instruction in the current session, the direct user instruction wins.
8. If a requested action would violate this contract, stop and explain why.

---

## Codex / GPT Rules

### Scope

- Codex works **only on the implementation branch**: `development`.
- Codex is the **only agent allowed to make routine code changes** by default.
- Codex may read files across the repository as needed for implementation, but must only edit files relevant to the assigned task.

### Allowed by default

- Implement tasks listed in `REVIEW_TASKS.md`
- Fix bugs
- Add or update tests
- Refactor code within the assigned scope
- Run targeted tests, lint, and build commands relevant to the touched area
- Commit small checkpoints when asked

### Not allowed by default

- Do not switch work to `master`
- Do not merge branches
- Do not rewrite the architecture without explicit approval
- Do not resolve review comments by changing unrelated code
- Do not silently ignore tasks in `REVIEW_TASKS.md`
- Do not edit this contract unless explicitly asked

### Required output after implementation work

Codex should report:

1. files changed
2. commands run
3. test/lint/build results
4. unresolved risks or follow-up items

---

## Claude / Sonnet Rules

### Scope

- Claude works **only on the review branch**: `master`.
- Claude is **review-only by default**.
- Claude should primarily inspect the diff between `master` and `development`.

### Allowed by default

- Review diffs
- Identify correctness risks
- Identify regression risks
- Identify missing tests
- Suggest minimal fixes
- Create or update `REVIEW_TASKS.md`
- Summarize review findings
- Produce merge-readiness checklists

### Not allowed by default

- Do not make code changes unless the user explicitly asks Claude to write code
- Do not perform feature implementation by default
- Do not edit files unrelated to review artifacts
- Do not merge branches
- Do not move implementation work from Codex onto Claude unless explicitly instructed

### Review method

Claude should review with this priority order:

1. correctness
2. regressions
3. missing validation/tests
4. maintainability of touched code
5. only then style / cleanup

Claude should prefer reviewing the current diff only, rather than re-analyzing the entire repository.

---

## Handoff Protocol

### Claude -> Codex

After review, Claude writes actionable items to `REVIEW_TASKS.md`.

Each task must:

- be specific
- map to an actual issue in the diff
- describe the expected fix
- include validation guidance when relevant
- avoid vague wording like "improve this" or "clean up"

### Codex -> Claude

After implementing review items, Codex should indicate:

- which tasks were completed
- which tasks were not completed
- why any task was deferred or rejected
- what validation was run

---

## REVIEW_TASKS.md Format

Claude should use this structure when writing review tasks:

```md
# Review Tasks

## Context
- Review branch: master
- Implementation branch: development
- Review date: YYYY-MM-DD

## Open Tasks

### TASK-001
Status: open
Priority: high
Area: <path or subsystem>
Problem: <what is wrong>
Required fix: <what Codex should change>
Validation: <tests / lint / manual repro>
Notes: <optional>

## Completed Tasks
<!-- move finished tasks here instead of deleting them immediately -->
```

Codex should update task statuses instead of rewriting the whole file when practical.

---

## Override Rules

The user may override any part of this contract in the current session.

Examples of valid overrides:

- "Claude may implement this fix directly."
- "Codex may review instead of changing code for this task."
- "Use branch `feature-x` instead of `development`."

Overrides should be interpreted narrowly. Do not treat a narrow override as a full suspension of the contract.

---

## Safe Defaults

If the instruction is ambiguous:

- Claude should review, not implement.
- Codex should implement only on `development`.
- Neither agent should merge.
- Ask for clarification only if the ambiguity blocks safe progress.

---

## Maintenance

Keep this file short, specific, and current.
When the implementation branch changes, update only:

- `Active Branch Roles`
- `REVIEW_TASKS.md Format` branch references if needed

Do not add project trivia or long style guides here. Put broader coding conventions elsewhere.
