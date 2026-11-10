# Agent instructions for Tsim

These notes are for agents working on Tsim's code. To use Tsim in other projects, read the
user skills in `skills/` instead (`simulating-circuits-with-tsim`, `running-qec-experiments-with-tsim`, `porting-circuits-to-tsim`); they
also explain the cost model (active width) and the conventions users rely on.

## Commands

```bash
uv sync                          # environment, including the dev group
uv run pytest test               # full suite, about 90 s; test/unit alone is faster
uv run pre-commit run --all-files  # black, isort, ruff, pyright, codespell, deptry, nbstripout
just doc-build                   # docs build and Mintlify checks (needs `npm i -g mint`)
```

`test/unit/utils/test_kernels.py::test_the_calling_thread_keeps_parallel_kernels_after_run_tasks`
fails on hosts where numba uses the TBB threading layer; that failure does not come from your
change.

## How Tsim works

A circuit goes through these stages; each module's docstring says more.

1. `circuit.py`: `Circuit` wraps a `stim.Circuit`. Non-Clifford gates are stored as tagged
   Clifford instructions (`T` as `S[T]`, `R_Z(a)` as `I[R_Z(theta=a*pi)]`), see `core/tags.py`.
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
- `test/unit/test_skills.py` runs the code in the user skills; update the skills when you
  change behavior they describe.

## Performance

- Benchmark a change against `main` on the same machine, with several circuit families
  (cultivation, distillation, coherent-noise surface codes, quantum volume), not one.
- Numba caches compiled kernels on disk (`utils/kernels.py`). Measure with a warm cache in a
  fresh process as well as in-process: cached kernels have behaved differently from freshly
  compiled ones (passing large structs by value was 5 to 8 times slower when cached). After
  changing a jitted function's signature or return type, delete stale `*.nbi`/`*.nbc` files.
- Prefer one general solution over special cases for particular benchmarks, and drop
  optimizations whose gain is small compared with the code they add.

## Running heavy jobs

Agents share this machine with people and other agents.

- Run long or memory-hungry jobs under a hard cap, for example
  `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p CPUQuota=400% timeout 1800 <cmd>`.
  A limit stated only in a prompt is not enough: an uncapped job once used 55 GB and the
  out-of-memory killer took down other sessions.
- Check `sampler.active_width` before sampling anything wide, and set `NUMBA_NUM_THREADS` or
  `numba.set_num_threads` to the cores you were given.
- Never `pkill -f` or `pgrep -f` a pattern that also appears in your own command line; it
  matches your own shell.

## Style

- Write "Tsim" and "Stim" in prose (comments, docstrings, docs, changelog); lower case only in
  code.
- Google-style docstrings (ruff `D` rules). Explain non-obvious choices in `# NOTE:` comments.
- Keep changes small and simple; remove code that a change makes unnecessary instead of
  leaving it behind.

## Pull requests

- Branch off `main`; use [Conventional Commits](https://www.conventionalcommits.org/).
- Add a `CHANGELOG.md` entry under `## [Unreleased]` for any user-visible change.
- Bug fixes include a minimal reproducer as a test; performance changes include before and
  after numbers, with the machine, thread count and circuits.
- Run the full test suite and pre-commit before asking for review.
