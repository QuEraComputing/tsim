---
name: shipping-tsim-changes
description: Takes a change to Tsim's code from a branch to a reviewed local commit, with the checks, evidence and git rules this repository uses, scaled to what the change touches. Use when starting a feature, bug fix or optimization in the Tsim repository, when updating an existing pull request, or before saying that a change is done.
metadata:
  internal: true
---

# Shipping a change to Tsim

## 1. Start

**New work.** Work in a worktree on a new branch off the newest main, not in the main
checkout: `git fetch origin main && git worktree add -b <type>/<topic> ../tsim-<topic>
origin/main`, then `uv sync` there. Stack on another branch only when the change needs it.

**Updating an existing pull request** (review comments, a follow-up, a fix the user asks
for on that PR). Keep its branch: find it with `gh pr view <n> --json headRefName`, use
the worktree that already has it (`git worktree list`) or add one for that branch, and
commit on top. Do not start a new branch or a new PR for it. Rebase only when the user
asks or the branch conflicts with its base; a rebase rewrites a pushed branch, so say
that it needs a force-push.

**Acceptance criteria.** Infer them from the task and the code, and state them in a line
or two before you start: the behavior, the test that shows it, and for performance work
the circuits and the gain that matters. Ask only when they are unclear, for example an
unstated performance target, two plausible behaviors, or a change to public API.

## 2. Build

- For a bug, write the failing test first and watch it fail.
- `uv run tsim-dev test --changed` while iterating.
- Anything long or memory-hungry goes through `uv run tsim-dev run -- <command>` or
  `uv run tsim-dev job start NAME -- <command>`; see AGENTS.md.

## 3. Verify

Run what applies and keep the output. A claim without the command and its result is not
evidence.

| Change | Required |
|---|---|
| Any | `uv run tsim-dev check` prints PASS for lint and tests |
| Could affect compile or sampling cost (see below) | `uv run tsim-dev bench --ref origin/main`; any regression investigated and explained |
| Aims to make Tsim faster | the same with `--suite full`; see the `benchmarking-tsim` skill |
| Could change the active width (graph, simplification, scheduling, plan) | `uv run tsim-dev bench --width --ref origin/main`; any `wider` explained |
| Bug fix | the new test fails on `origin/main` and passes on the branch |
| User-visible | a `CHANGELOG.md` entry under `## [Unreleased]` |
| Behavior the user skills describe | `skills/` updated; `test/unit/test_skills.py` passes |

**When to benchmark.** Code on the compile path (parsing, the ZX graph and its
simplification, components, the plan, scheduling, lowering to the action tape) or the
sampling path (samplers, noise, the state-vector executor, numba kernels), and changes
to dependencies or numba settings, can change performance. Docs, tests, type hints,
error messages, printing and code that runs once per call outside those paths do not.
When unsure, run the quick suite: it takes a few minutes.

**A regression is a finding, not a veto.** If the benchmark reports `slower` (exit 1),
do not stop at the exit code and do not hide it:

1. Confirm it: rerun the affected circuits with `--rounds 9 --min-seconds 1`.
2. Find the cause: `tsim-dev profile` on one affected circuit, on both checkouts.
3. Remove it if the change does not need it. If it does, for example a correctness fix
   that has to do more work, keep it and say so: which circuits, by how much, and why.
   A wrong result is never the faster option.

Then read your own diff (`git diff origin/main --stat`, then the diff). Remove anything
the task did not need: special cases for one benchmark, leftover debug code, unrelated
reformatting. Ask whether a simpler change would do.

## 4. Commit and review

- Commit locally with a Conventional Commits message.
- Do not push, open a pull request, force-push or merge unless the user asked for that
  step in this conversation. Some branches are not meant to be public yet. When updating
  an existing PR, pushing to its branch still needs the user's go-ahead.
- For a review, give a fresh subagent the diff and the task, not your conclusions, and
  ask for correctness findings only. Fix what it finds, then run step 3 again.

## 5. Report

Say in a few sentences what changed, the verification commands you ran with their
results, and what you did not verify. For performance-relevant changes include the bench
table, the CPU model and the thread count, and the explanation of any regression. Say
plainly if something failed or was skipped.
