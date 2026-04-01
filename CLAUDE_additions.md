@AGENTS.md

## Claude-specific reinforcement

- Default mode in this repository is **review-only**.
- Unless the user explicitly tells Claude to write code, Claude should restrict itself to:
  - reviewing `master...v9.1`
  - updating `REVIEW_TASKS.md`
  - summarizing risks and missing tests
- If implementation is requested, keep the change as small as possible and state clearly that this is an explicit override of the default contract.
