# Skill evals

The user skills in `skills/` are tested at three levels:

| Level | What it checks | Where it runs |
|---|---|---|
| Static | Frontmatter follows the [Agent Skills spec](https://agentskills.io/specification), links resolve, every Python block runs against the current Tsim | CI, `pytest test/unit/test_skills.py` |
| Triggering | The right skill loads for a prompt, and none loads for near-misses | Locally, `run.py triggers` |
| Behavior | Agents do realistic tasks better with the skills than without | Locally, `run.py cases` |

Triggering and behavior evals call a real agent (Claude Code or Codex CLI) on your account, so
they cost money and give slightly different results on every run. Run them before changing a
skill's description or advice, and after, and compare.

## Running

From the repository root, with the dev environment synced (`uv sync`):

```bash
# Which skill loads for each prompt in triggers.json (about $0.2 per query with Claude)
python evals/skills/run.py triggers
python evals/skills/run.py triggers --only 0 --only 13      # a quick subset

# Each case with and without the skills (about $0.3 to $2 and 1 to 10 minutes per run)
python evals/skills/run.py cases
python evals/skills/run.py cases --case cultivation-feasibility --runs 3

# Use Codex instead of Claude, or pin a model
python evals/skills/run.py cases --agent codex
python evals/skills/run.py triggers --model claude-sonnet-5-5

# Regrade a results directory after editing a check
python evals/skills/run.py grade evals/skills/results/<dir>
```

Results go to `evals/skills/results/<timestamp>-<kind>-<agent>/` (ignored by git): a
`summary.md`, and per run the agent's working directory, transcript, final answer and grades.

## How a run works

- Each run gets a fresh directory with the case's `inputs/` and a `python` on `PATH` that
  runs this repository's environment. In the `with` arm, the skills are copied to
  `.claude/skills/` (Claude Code) and `.agents/skills/` (Codex); the `without` arm has none.
- The agent runs headless (`claude -p` or `codex exec`) under `systemd-run` with a memory cap
  (`--mem`, default 8G), a CPU cap (`--cpus`, default 4) and a timeout. Agents run real
  simulations, and an uncapped one once used 55 GB by trying a stabilizer-rank strategy; with
  the cap only that process is killed. `--no-cap` disables the limits.
- A skill counts as loaded when Claude calls its `Skill` tool or either agent reads a file under
  `skills/<name>/`.
- `cases/<name>/check.py` grades the final answer, the code the agent ran or wrote, and the
  files it left behind. Checks are deterministic: known values (active width, acceptance,
  logical error rate) and things the skills teach (detector sampler, postselection inside Tsim,
  no stabilizer-rank strategy, a decomposed model for PyMatching). Shared checks are in
  `checks.py`.

Codex runs shell commands in its own sandbox (`--codex-sandbox`, default `workspace-write`).
That sandbox cannot start inside another sandbox, for example when Codex is launched from
within a Claude Code session; run Codex evals from a normal terminal.

## Adding a case

1. Create `cases/<name>/case.toml` with the `skill` the case targets and a realistic `prompt`,
   and put the files it needs in `cases/<name>/inputs/`.
2. Write `cases/<name>/check.py` with `check(answer, transcript, work) -> dict`, mapping each
   check's name to `(passed, evidence)`. Prefer checks that a run without the skill can fail;
   a check that always passes does not tell the arms apart.
3. Check the graders against a few real answers (`run.py grade`) before trusting a score.

To add a triggering query, append `{"skill": "<name>" or null, "query": "..."}` to
`triggers.json`. Near-misses that should load no skill are as useful as positive cases.
