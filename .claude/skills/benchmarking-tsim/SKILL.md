---
name: benchmarking-tsim
description: Measures Tsim throughput and compile time with `tsim-dev bench` on the tsim-bench circuits and compares a change against a reference commit or checkout. Use when asked to benchmark Tsim, to check a change for performance regressions, to produce before and after numbers for a pull request, to run the tsim-bench corpus, or to add a benchmark circuit.
metadata:
  internal: true
---

# Benchmarking Tsim

Use `uv run tsim-dev bench`. Do not write timing scripts: hand-written ones have
measured the wrong thing often enough (compile time included, a cold numba cache,
several threads, a busy machine) that their numbers are not accepted in reviews.

## Check a change against main

```bash
git fetch origin main
uv run tsim-dev bench --ref origin/main        # quick suite, about 4 minutes
```

Use `origin/main`, not `main`: the local branch is often behind. `--ref` also takes
any commit, or the path of another checkout that has a `.venv`
(`--ref ../tsim-other-worktree`).

```
circuit                     k  shots/s   kept  compile   shots/s vs ref (min, max)      compile vs ref
inject_cultivate_d5_p0.001  6    13.9M  0.141     0.13   1.20x [1.19, 1.21] faster  0.96x same
```

- `kept` is the fraction of shots that survive postselection (`-` when there is none).

- `shots/s vs ref` is new over reference, so above 1 is faster. The bracket is the
  smallest and largest ratio over the rounds.
- A verdict of `faster` or `slower` needs the mean outside the tolerance (5%) and
  every round on the same side of 1. `noisy` means the rounds disagree: run again
  with `--rounds 9 --min-seconds 1` before you conclude anything.
- `kept-differs`: the two sides kept different fractions of shots (beyond binomial
  noise), so they did not sample the same thing and the rate is not compared. It
  counts as a failure: a change to postselection or to the sampled distribution.
- Exit code 0: nothing slower. 1: at least one `slower` in rate or compile time, or
  a `kept-differs`.
  2: a circuit could not be measured.
- The JSON under `.tsim-dev/bench/` has every round.

## Active width only (`--width`)

For changes to compilation, scheduling or the plan, compare widths without sampling:

```bash
uv run tsim-dev bench --width --ref origin/main      # whole corpus, compile only
uv run tsim-dev bench --width --suite full --ref origin/main
```

Each side compiles every circuit once, in one process, and prints the width next to
the one recorded in tsim-bench. The verdict is `wider`, `narrower`, `same` or
`unknown`; any `wider` exits 1. A circuit over Tsim's limit shows `>=N`: N is the width
in Tsim's refusal, a lower bound. Widths do not depend on the machine, so this needs
no rounds or quiet machine, but it says nothing about speed: a change that keeps the
width can still be slower. Compile times are shown for each side.

## Compare settings: threads, strategy

`--side REF[:key=value,...]`, given twice, replaces `--ref` when the two sides differ in
more than the checkout. REF is `.` (this checkout), a git revision, a checkout path, or
a Python executable (an environment built elsewhere, such as one with the GPU extra).
Keys: `threads=N`, `device=cpu|gpu`, `strategy=NAME`. The first side is "new".

```bash
uv run tsim-dev bench --side .:threads=10 --side .:threads=1          # 10 cores vs 1
uv run tsim-dev bench --side $GPU_PY:device=gpu --side $GPU_PY:device=cpu
uv run tsim-dev bench --side .:threads=4 --side origin/main:threads=4
```

- `device=cpu` hides CUDA devices; `device=gpu` fails the circuit unless the worker finds
  Tsim's CUDA executor usable. Circuits above the executor's width limit still fall back
  to the CPU on that side, so check the width before reading a GPU ratio.
- Compile time is compared too; across devices a "slower" compile verdict (and exit code
  1) is expected, so read the rate column.

## What the command controls for you

- One thread, pinned to a core (`--threads N` to change; say so when you report).
- Each measurement is a fresh process with a warm numba cache, under a memory cap.
  Kernels loaded from the cache have behaved differently from freshly compiled
  ones, and the cached case is what users get.
- Compile time and the first call are excluded from shots/s and reported apart.
- The two checkouts alternate round by round, so drift affects both.
- Circuits come from tsim-bench (https://github.com/rafaelha/tsim-bench) at the commit
  pinned in `dev/tsim_dev/corpus.py`, fetched into `~/.cache/tsim-dev` on first use.
  Each circuit's manifest there records its arXiv paper, its sampler and how the paper
  postselects; the benchmark postselects that way by default. `--no-postselect`
  samples every shot instead. shots/s counts requested shots, kept or not.

## What you must do

- Run one benchmark at a time, and nothing else heavy (tests, other agents' jobs)
  next to it. Heed the busy-machine warning.
- A change must hold across families. Run `--suite full` before claiming a speedup,
  and report the circuits that got slower as well as the ones that got faster.
- For a single circuit: `uv run tsim-dev bench inject_cultivate_d5_p0.001 --ref origin/main`
  (a file stem or a full id from `tsim-dev corpus list`), or a path to any `.stim` file
  with `--sampler` and `--postselect all|none`.
- Postselection changes what is measured: a change that speeds up rejected shots shows
  only with it on. Report which you used, and run both when the change touches the
  postselection path.
- Paste the table in the pull request, with the CPU model (`lscpu | grep 'Model name'`)
  and the thread count.

## Suites and the whole corpus

Suites are defined in tsim-bench's `suites.toml`: `quick` (the default, a few minutes),
`full` (one or two circuits per family, widths up to about 16) and `hard` (widths 16 to
26, such as the tricycle CCZ factories; minutes per circuit). `--suite corpus` is every
circuit, over 190 from 28 papers, so run it as a job:

```bash
uv run tsim-dev corpus list --max-width 12      # what would run
uv run tsim-dev job start corpus --timeout 14400 -- \
    uv run tsim-dev bench --suite corpus --max-width 12 --ref origin/main
uv run tsim-dev job wait corpus                 # repeat until it reports the end
```

`--max-width K` skips circuits whose recorded active width is larger than K, and those
with no recorded width; `--width-of clifft` (or another backend in tsim-bench's
`k_max.json`) filters on that backend's width instead, for backend work that targets a
different regime. `--only SUBSTRING` keeps circuits whose id contains it.

## Limits

- Tsim only, on the CPU or (with `--side ...:device=gpu`) its CUDA executor.
  Comparisons with other simulators (Clifft, SymFT, Magna) are run from the
  tsim-bench repository, which pins each simulator's environment.
- To add a circuit or change a suite, change tsim-bench: put the file in its paper's
  folder, add an entry to that folder's `circuits.toml` (arXiv id, sampler, postselection
  and the reason for it, see its `BENCHMARKS.md`), run `python -m tsim_bench check`
  there, then bump `REV` in `dev/tsim_dev/corpus.py`. Until the commit is pushed, point
  the harness at the checkout with `--corpus-dir` or `TSIM_BENCH_DIR`.
