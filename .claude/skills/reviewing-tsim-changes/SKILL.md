---
name: reviewing-tsim-changes
description: Runs the review of a Tsim change before it merges, with the number of reviewers and rounds set by `tsim-dev review plan`, findings that must come with a failing test, and a merge gate that checks bot comments. Use when asked to review a branch or pull request in the Tsim repository, to get a change ready to merge, to handle Copilot or Codex review comments, or to merge a pull request.
metadata:
  internal: true
---

# Reviewing a Tsim change

The owner merges without reading the code. What stands in for reading is this
procedure, so do not shorten it and do not add to it: extra rounds on a small diff
have produced only nits here, and missing rounds on a large one have let bugs out.

## 1. Find out what the change owes

```bash
uv run tsim-dev review plan       # tier, reasons, required steps
uv run tsim-dev review status     # the same, checked; exit 0 ready, 1 to do, 3 waiting
```

| Tier | When | Reviews |
|---|---|---|
| 0 | no change in `src/tsim` | none |
| 1 | other changes in `src/tsim` | one, in a fresh context |
| 2 | sampling code (`statevector/`, `noise/`, `compile/`, `sampler.py`, kernels) or more than 300 changed source lines | two, from different model families, then rounds until one has no findings; at most 3 rounds |

Benchmarks do not depend on the tier: `plan` asks for `tsim-dev bench --ref <base>`
when the diff touches the compile or sampling path or the dependencies, and for
`bench --width` when it can change the active width. A benchmark that finds a
regression still counts as run; `status` notes the regression, and it has to be
confirmed, profiled and explained in the pull request (see the
`shipping-tsim-changes` skill).

Notes printed by `plan`: `SPLIT` (over 600 source lines: split into stacked pull
requests before reviewing), `HUMAN` (tell the user which files they should read),
`TESTS` (explain removed tests).

## 2. Run a review round

Start the reviewer as a subagent with a fresh context (`tsim-reviewer` in
`.claude/agents/`). For the second family in tier 2, use the model the user names,
or ask which one is available. Give the reviewer:

- the command for the diff (`git diff <merge-base>...HEAD`), and for a later round
  only the diff since the last reviewed commit;
- the task: the issue or the user's request, in their words;
- the findings already known, so that it does not repeat them.

Do not give it the pull request description, your summary, or your opinion of the
change. Reviewers that read the author's account find fewer bugs.

## 3. Accept only findings with a failing test

A finding counts when the reviewer gives a test or a short script that fails on the
branch, with the expected and the observed result. For each one:

1. Run the reproducer yourself. If it does not fail, drop the finding.
2. Add it to the test suite, fix the code, and show the test pass.
3. For a performance finding, the evidence is `tsim-dev bench`.

Do not act on unconfirmed findings, style remarks or suggested abstractions. Do not
weaken or delete a test to make a finding go away.

Then record the round, with the number of confirmed findings:

```bash
uv run tsim-dev review record --model opus-5.5 --findings 2
```

If round 3 of a tier 2 change still has findings, stop. Tell the user that the
change does not converge and propose a split or another design.

## 4. Bot comments on the pull request

Copilot and Codex review automatically within about five minutes of a push; of
their comments here, about half pointed at real bugs and almost none were wrong.
`review status` lists every thread that has no reply.

- Real: fix it, with a test where one is possible, and reply with the commit.
- Not real or out of scope: reply with the reason in one sentence. For something
  real but out of scope, open an issue and link it.
- Never merge with unanswered threads, and wait for reviewers that `status` reports
  as pending.

## 5. Merge

Only when the user asked for the merge, and only as

```bash
uv run tsim-dev review status && gh pr merge <number> --squash
```

from the pull request's branch. A hook blocks `gh pr merge` in any other form. In
your report, paste the output of `review status`, and list anything it marked as a
note (files for a person to read, removed tests).
