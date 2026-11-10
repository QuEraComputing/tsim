---
name: shipping-tsim-changes
description: Takes a change to Tsim's code from a new branch to a reviewed local commit, with the checks, evidence and git rules this repository uses. Use when starting a feature, bug fix or optimization in the Tsim repository, when asked to open or update a pull request, or before saying that a change is done.
---

# Shipping a change to Tsim

## 1. Start

- Work in a worktree on a new branch off the newest main, not in the main checkout:
  `git fetch origin main && git worktree add -b <type>/<topic> ../tsim-<topic> origin/main`,
  then `uv sync` there. Stack on another branch only when the change needs it.
- Agree on what "done" means before writing code: the behavior, the test that shows
  it, and for performance work the circuits and the size of the gain that matters.

## 2. Build

- For a bug, write the failing test first and watch it fail.
- `uv run tsim-dev test --changed` while iterating.
- Anything long or memory-hungry goes through `uv run tsim-dev run -- <command>` or
  `uv run tsim-dev job start NAME -- <command>`; see AGENTS.md.

## 3. Verify

Run these and keep their output. A claim without the command and its result is not
evidence.

| Change | Required |
|---|---|
| Any | `uv run tsim-dev check` prints PASS for lint and tests |
| Touches `src/tsim` outside docstrings | `uv run tsim-dev bench --ref origin/main` exits 0 |
| Performance | the same with `--suite full`; see the `benchmarking-tsim` skill |
| Bug fix | the new test fails on `origin/main` and passes on the branch |
| User-visible | a `CHANGELOG.md` entry under `## [Unreleased]` |
| Behavior the user skills describe | `skills/` updated; `test/unit/test_skills.py` passes |

Then read your own diff (`git diff origin/main --stat`, then the diff). Remove
anything the task did not need: special cases for one benchmark, leftover debug
code, unrelated reformatting. Ask whether a simpler change would do.

## 4. Commit and review

- Commit locally with a Conventional Commits message.
- Do not push, open a pull request, force-push or merge unless the user asked for
  that step in this conversation. Some branches are not meant to be public yet.
- For a review, give a fresh subagent the diff and the task, not your conclusions,
  and ask for correctness findings only. Fix what it finds, then run step 3 again.

## 5. Report

Say in a few sentences what changed, the verification commands you ran with their
results, and what you did not verify. For performance work include the bench table,
the CPU model and the thread count. Say plainly if something failed or was skipped.
