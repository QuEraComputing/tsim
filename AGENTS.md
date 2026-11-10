# Agent instructions for Tsim

These notes are for agents working on Tsim's code. To use Tsim in other projects, read the
user skill in `skills/simulating-circuits-with-tsim/` instead; it also explains the cost model
(active width) and the conventions users rely on.

## Commands

`uv sync` installs everything, including the `tsim-dev` CLI (`dev/`). Use it instead of
hand-written timing scripts, `nohup`, sleep loops and bare `pytest`: it caps memory, keeps
output short, writes details to `.tsim-dev/`, and its exit codes mean 0 pass, 1 the code
failed (or got slower), 2 the harness could not run, 3 still running.

```bash
uv run tsim-dev check                   # the gate before review: pre-commit, full suite (~90 s)
uv run tsim-dev test --changed          # tests of the changed files; or give paths
uv run tsim-dev bench --ref origin/main # quick A/B benchmark, ~1 min; exit 1 if slower
uv run tsim-dev bench --width --ref origin/main  # active widths only, nothing sampled
uv run tsim-dev inspect FILE.stim       # qubits, T count, active width, stats; no sampling
uv run tsim-dev profile cultivation_d5  # where compile and sampling time goes (py-spy)
uv run tsim-dev run -- CMD              # any command under memory, CPU and time caps
uv run tsim-dev job start NAME -- CMD   # the same in the background; then `job wait NAME`
just doc-build                          # docs build and Mintlify checks (needs `npm i -g mint`)
```

`--help` on each subcommand lists the caps and options.

## Skills for working on Tsim

`.claude/skills/` (also at `.agents/skills/`) holds procedures for this repository; they
are not published to users.

- `shipping-tsim-changes`: branch, verify, commit, review; what evidence a change needs.
- `benchmarking-tsim`: A/B benchmarks, reading verdicts, the tsim-bench corpus.
- `profiling-tsim`: sizing a circuit, reading profiles, choosing what to optimize.
- `testing-tsim-correctness`: oracles and statistical tests for sampling changes.

## How Tsim works

A circuit goes through these stages; each module's docstring says more.

1. `circuit.py`: `Circuit` wraps a `stim.Circuit`. Non-Clifford gates are stored as tagged
   Clifford instructions (`T` as `S[T]`, `R_Z(a)` as `I` with a parametric tag); `core/tags.py`
   reads and writes the tags.
2. `core/parse.py`, `core/builder.py`: the circuit becomes a ZX graph (zxparam), with Pauli
   noise as parameterized vertices.
3. `core/graph.py`: `prepare_graph` uses detectors and observables (detector sampler) or
   measurement records (measurement sampler) as outputs, runs `full_reduce`, and changes the
   noise to an error basis (`transform_error_basis`).
4. `compile/pipeline.py`: each connected component becomes either a *direct* output (one
   error-basis bit or a constant, sampled from the noise alone) or a quantum component.
5. `statevector/`: for the default `strategy="active_statevector"`, each quantum component is
   read off as Pauli rotations and measurements (`pbc.py`), reordered to keep the state small
   (`schedule.py`), turned into a fixed-width plan and an action tape (`plan.py`,
   `program.py`), and run by numba kernels (`shots.py`, `shared.py`, with trajectory sharing).
6. `noise/channels.py` samples the noise; `sampler.py` holds the public samplers.

The stabilizer-rank strategies (`cat5`, `bss`, `cutting`, in `compile/` with JAX) are
provisional and may be deprecated. Do not extend them, and never run them on circuits with
more than a handful of T gates: their compile memory grows exponentially.

## Invariants

- **Public API:** `Circuit`, `CompiledMeasurementSampler` and `CompiledDetectorSampler` follow
  semantic versioning (`tsim/__init__.py`, `test/unit/test_public_api.py`); `tsim.experimental`
  may change in any release; every other module is internal.
- **Raw outputs:** detectors and observables are raw parities, unlike Stim's flips relative to
  a reference. The reference-sample flags change only that.
- **Seeds:** with `active_statevector`, a seed's samples do not depend on the thread count or
  `batch_size`. Do not break this; noise blocks and per-shot draws are seeded by call and shot
  index (`shots.py`).
- **Direct outputs stay direct:** outputs that depend on the noise alone never go through a
  state vector. Most detectors of QEC circuits are direct, which is why the detector sampler is
  fast.
- **Detector error models** come from Stim, applied to the Clifford stand-in
  (`Circuit.detector_error_model`); they are only guaranteed correct for Clifford circuits
  (see issues #226 and #247).

## Testing

- Test behavior and invariants: compare against dense references (`test/helpers/statevector.py`,
  `test/helpers/tensors.py`), against Stim on Clifford circuits and stand-ins (with the
  reference-sample flags), and against exact values where they exist.
- A test of a fix must fail without the fix. Keep circuits small, so tests stay fast and do not
  build large tensors.
- Statistical tests need enough shots for the tolerance, and a fixed seed.
- `test/unit/test_skills.py` runs the code in the user skill; update the skill when you change
  behavior it describes.

## Performance

- Measure with `tsim-dev bench --ref origin/main`, never with a script of your own. It runs
  fresh processes on one pinned thread with a warm numba cache, excludes compile time, and
  interleaves the two checkouts. A speedup must hold on `--suite full`, across families.
- The bench times bit-packed rows from the sampler that tsim-bench records (the detector
  sampler for QEC circuits), on one CPU thread. The sampler, boolean output, the thread
  count and a GPU each change the rate severalfold; the `profiling-tsim` skill says how
  to measure each, and a report names the ones it used.
- Benchmark changes that could affect compile or sampling cost (see the
  `shipping-tsim-changes` skill), not every change. A regression is investigated and
  explained, not hidden; a correctness fix may cost performance if the cost is stated.
- `tsim-dev bench --width --ref origin/main` compares active widths without sampling, for
  changes to the graph, simplification, scheduling or the plan.
- Numba caches compiled kernels on disk (`utils/kernels.py`). Cached kernels have behaved
  differently from freshly compiled ones (passing large structs by value was 5 to 8 times
  slower when cached), which is why the benchmark uses the cache. After changing a jitted
  function's signature or return type, delete stale `*.nbi`/`*.nbc` files.
- Run one benchmark at a time and nothing heavy beside it.
- Prefer one general solution over special cases for particular benchmarks, and drop
  optimizations whose gain is small compared with the code they add.

## Running heavy jobs

Agents share this machine with people and other agents.

- Run anything long or memory-hungry through `tsim-dev run` or `tsim-dev job start`
  (defaults: 8 GB, 4 CPUs, no swap). A limit stated only in a prompt is not enough: an
  uncapped job once used 55 GB and the out-of-memory killer took down other sessions. A
  hook in `.claude/settings.json` blocks uncapped background Python and `pkill -f`.
- Wait for a job with `tsim-dev job wait NAME`, not a sleep loop; stop it with
  `tsim-dev job stop NAME`. `pkill -f` matches your own shell and kills it.
- Check the active width (`tsim-dev inspect`) before sampling anything wide; memory per
  shot grows like 2^k.

## Style

- Write "Tsim" and "Stim" in prose (comments, docstrings, docs, changelog); lower case only in
  code.
- Google-style docstrings (ruff `D` rules). Explain non-obvious choices in `# NOTE:` comments.
- Keep changes small and simple; remove code that a change makes unnecessary instead of
  leaving it behind.

## Pull requests

- New work: branch off `origin/main` in a worktree. Updating an existing pull request: keep
  its branch and commit on top. Use [Conventional Commits](https://www.conventionalcommits.org/).
- Commit locally. Push, open, force-push or merge a pull request only when asked.
- Add a `CHANGELOG.md` entry under `## [Unreleased]` for any user-visible change.
- Bug fixes include a minimal reproducer as a test that fails without the fix;
  performance-relevant changes include the `tsim-dev bench` table, the CPU model, the
  thread count and the explanation of any regression.
- `tsim-dev check` must pass before review. Report the commands you ran and their results,
  and say what you did not verify.
