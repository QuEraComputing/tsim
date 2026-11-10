---
name: benchmarking-tsim
description: Measures Tsim throughput and compile time with `tsim-dev bench` and compares a change against a reference commit or checkout. Use when asked to benchmark Tsim, to check a change for performance regressions, to produce before and after numbers for a pull request, or to run the tsim-bench circuit corpus.
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
circuit          k  shots/s  compile   shots/s vs ref (min, max)      compile vs ref
cultivation_d5   6    13.9M     0.13   1.20x [1.19, 1.21] faster  0.96x same
```

- `shots/s vs ref` is new over reference, so above 1 is faster. The bracket is the
  smallest and largest ratio over the rounds.
- A verdict of `faster` or `slower` needs the mean outside the tolerance (5%) and
  every round on the same side of 1. `noisy` means the rounds disagree: run again
  with `--rounds 9 --min-seconds 1` before you conclude anything.
- Exit code 0: nothing slower. 1: at least one `slower` in rate or compile time.
  2: a circuit could not be measured.
- The JSON under `.tsim-dev/bench/` has every round.

## What the command controls for you

- One thread, pinned to a core (`--threads N` to change; say so when you report).
- Each measurement is a fresh process with a warm numba cache, under a memory cap.
  Kernels loaded from the cache have behaved differently from freshly compiled
  ones, and the cached case is what users get.
- Compile time and the first call are excluded from shots/s and reported apart.
- The two checkouts alternate round by round, so drift affects both.
- Each registry circuit fixes its sampler and postselection
  (`dev/circuits/circuits.toml`). With postselection, shots/s counts attempted shots.

## What you must do

- Run one benchmark at a time, and nothing else heavy (tests, other agents' jobs)
  next to it. Heed the busy-machine warning.
- A change must hold across families. Run `--suite full` before claiming a speedup,
  and report the circuits that got slower as well as the ones that got faster.
- For a single circuit: `uv run tsim-dev bench cultivation_d5 --ref origin/main`, or
  a path to any `.stim` file with `--sampler` and `--postselect`.
- Paste the table in the pull request, with the CPU model (`lscpu | grep 'Model name'`)
  and the thread count.

## The tsim-bench corpus

189 circuits from 27 papers (https://github.com/rafaelha/tsim-bench). It takes much
longer, so run it as a job:

```bash
uv run tsim-dev corpus fetch                    # once; or set TSIM_BENCH_DIR to a clone
uv run tsim-dev job start corpus --timeout 14400 -- \
    uv run tsim-dev bench --corpus --max-k 12 --ref origin/main
uv run tsim-dev job wait corpus                 # repeat until it reports the end
```

`--only SUBSTRING` selects circuits by name; `--max-k` skips circuits whose recorded
active width is larger.

## Limits

- CPU only. GPU runs and comparisons with other simulators (Clifft, SymFT, Magna)
  are run from the tsim-bench repository, which pins each simulator's environment.
- To add a circuit to the registry, put the file in `dev/circuits/`, add it to
  `circuits.toml` and `git add -f` it (`*.stim` is ignored). Keep the quick suite
  to one small circuit per family.
