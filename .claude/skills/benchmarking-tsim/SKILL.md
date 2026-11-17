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
uv run tsim-dev bench --ref origin/main        # quick suite, about a minute
```

Use `origin/main`, not `main`: the local branch is often behind. `--ref` also takes
any commit, or the path of another checkout that has a `.venv`
(`--ref ../tsim-other-worktree`).

```
circuit                     k device  shots/s   kept  compile   shots/s vs ref (min, max)      compile vs ref
inject_cultivate_d5_p0.001  6    cpu    13.9M  0.141     0.13   1.20x [1.19, 1.21] faster  0.96x same
```

- `device` is where the new side sampled (`cpu`, or a GPU backend: `metal` or `cuda`).
  A row that ends in `DEVICE: cpu -> metal` (or `cuda`) compares two devices: its ratio is not
  a code change.
- `kept` is the fraction of shots that survive postselection (`-` when there is none).
- `k` is the active width of the compiled sampler. With `--ref`, a row that ends in
  `WIDER: k 6 -> 7` or `NARROWER: ...` means the two sides compiled different widths;
  `wider` fails the run. A width that changes between runs of one side is an error.

- `shots/s vs ref` is new over reference, so above 1 is faster. The bracket is the
  smallest and largest ratio over the rounds.
- A verdict of `faster` or `slower` needs the mean outside the tolerance (5%) and
  every round on the same side of 1. `noisy` means the rounds disagree: run again
  with `--thorough --rounds 9 --min-seconds 1` before you conclude anything.
- `kept-differs`: the two sides kept different fractions of shots (beyond binomial
  noise), so they did not sample the same thing and the rate is not compared. It
  counts as a failure: a change to postselection or to the sampled distribution.
- Exit code 0: nothing slower. 1: at least one `slower` in rate or compile time, a
  `kept-differs`, or a wider active width. 2: a circuit could not be measured.
- The JSON under `.tsim-dev/bench/` has every round.

## Active width only (`--width`)

For changes to compilation, scheduling or the plan, compare widths without sampling:

```bash
uv run tsim-dev bench --width --ref origin/main      # whole corpus, compile only
uv run tsim-dev bench --width --suite full --ref origin/main
```

Both sides compile every circuit once, at the same time, in `--jobs` workers (default
4). A row is printed as soon as a circuit is done on both sides, small circuits first,
with an `[n/N]` count, so the first rows appear within seconds; the row shows the width
next to the one recorded in tsim-bench. The verdict is `wider`, `narrower`, `same` or
`unknown`; any `wider` exits 1. `--case-timeout` (300 s) stops one slow compile;
`--timeout` (1800 s here) is the deadline for the whole run, after which the circuits
not yet compiled are reported as errors. A circuit over Tsim's limit shows `>=N`: N is the width
in Tsim's refusal, a lower bound. Widths do not depend on the machine, so this needs
no rounds or quiet machine, but it says nothing about speed: a change that keeps the
width can still be slower. Compile times are shown for each side.

## Compare settings: threads, strategy

`--side REF[:key=value,...]`, given twice, replaces `--ref` when the two sides differ in
more than the checkout. REF is `.` (this checkout), a git revision, a checkout path, or
a Python executable (an environment built elsewhere). Keys: `threads=N`,
`strategy=NAME`, `device=cpu|gpu`. The first side is "new".

```bash
uv run tsim-dev bench --side .:threads=10 --side .:threads=1          # 10 cores vs 1
uv run tsim-dev bench --side .:threads=4 --side origin/main:threads=4
```

- Compile time is compared too; when the sides differ only in threads, read the rate
  column.

## GPUs

### Apple-silicon Macs (Metal)

On a Mac with an M-series chip, the default strategy samples on the GPU through Metal
whenever every component has at most 11 active qubits (k_max <= 11). A plain `uv sync`
installs it. That is what users of a MacBook get, and it is much faster than the CPU: on an
M4 Pro, 26-96x one CPU thread and 4-19x all ten performance cores, bit-packed. End-to-end
cultivation d=5 ran at 125M shots/s, 19x the ten cores. Circuits without noise whose CPU
trajectories share well, such as qv_q10, run at the CPU's rate.

The bench compiles with `device="cpu"` by default, so on a Mac a plain run measures the CPU
only.
Say which you measured, and on a Mac measure both:

```bash
uv run tsim-dev bench --device gpu --ref REF   # a GPU change; REF must have the Metal backend
uv run tsim-dev bench --side .:device=gpu --side .:device=cpu,threads=10   # GPU vs all cores
```

- The `device` column must say `metal` on the GPU side. It says `cpu` where Metal falls
  back: a component wider than 11, a rotation branch below float32's resolution (an angle
  below about 6e-5 π), noise beyond the GPU tables, no GPU memory. Under the default
  `device="auto"` these fallbacks are silent; in a script of your own, compile with
  `device="metal"` to make them raise instead.
- `--device cpu` (the default) runs tsim-bench's worker through `dev/tsim_dev/on_device.py`,
  which compiles with `device="cpu"`; `--device gpu` leaves Tsim's `device="auto"`.
- The GPU's rate depends on call size much more than the CPU's: each call pays a fixed cost
  (setting up the launches, waiting for the GPU to finish), and each batch of up to 2^20 shots
  (2^22 on CUDA) is one launch. The bench calibrates calls to about 0.3 s, which is enough. A GPU rate measured
  with calls of 10^5 shots or fewer, or with a small `batch_size`, measures overhead, not
  sampling; never report one. Leave `batch_size` unset on the GPU; it can only lower the
  batch, and it changes a seed's GPU samples.
- A reference without the Metal backend samples on the CPU even with `--device gpu`; its
  rows end in `DEVICE: cpu -> metal`, and the ratio compares devices, not code.
- Compare the GPU with all CPU cores (`--threads N`, from the `NOTE` line), not one.
- Say whether rates are bit-packed (the default) or `--bools`, above all on a GPU: booleans
  cost it 2.3-3.2x on Metal on circuits with hundreds of detectors (surface_d7_r7: 757M
  packed, 237M booleans on an M4 Pro) and 7-8x on CUDA (surface_d7_r7: 1,168M packed, 159M
  booleans on an RTX 5090), against 1.0-1.8x on the CPU, so a boolean comparison
  understates the GPU's lead. Compare Tsim with other simulators on the same output format.
- The GPU's own time per kernel: set `tsim.statevector.metal.runtime.PROFILE = {}` before
  a `sample` call, which then runs each dispatch alone and adds its GPU seconds under the
  kernel's name (`noise`, `tape`, `gather_rows`, ...).

### NVIDIA GPUs (CUDA)

With CuPy installed (`uv sync --extra cuda`; CUDA 13, driver 580 or later), the default
strategy samples on an NVIDIA GPU under the same conditions as Metal, and `"auto"` tries CUDA
before Metal. A plain `uv sync` does not install CuPy, so check that the `device` column says
`cuda`. On an RTX 5090 the full corpus (k_max <= 11) ran at a median of 286x one CPU core,
bit-packed; end-to-end cultivation d=5 at about 1.6G shots/s.

Everything in the Metal section holds, with `cuda` for `metal`, except:

- `--ref` builds the reference with `uv sync --no-dev`, without CuPy, so its side samples on
  the CPU and its rows end in `DEVICE: cpu -> cuda`. For a GPU A/B, check the reference out
  in a worktree, `uv sync --extra cuda` there, and pass it as a side:
  `--side .:device=gpu --side /path/to/ref-worktree:device=gpu`.
- With `--threads 1`, all of a circuit's workers are pinned to one core, so compiles of a
  minute or more (distillation d17+, stability t100) can time out; pass `--timeout 3600`.
- For the GPU's own time per kernel, use Nsight Systems (`nsys profile`) or Nsight Compute;
  the 5090 needs a 2025 or later release.

## What the command controls for you

- One thread, pinned to a core (`--threads N` to change; say so when you report).
  The `NOTE` line gives the cores Tsim uses by default.
- Bit-packed output (`--bools` for `sample()`'s default booleans) and the CPU
  (`device="cpu"`; `--device gpu` lets Tsim use a GPU backend). The header says
  which; the `profiling-tsim` skill lists how much each changes the rate.
- The sampler that tsim-bench records for each circuit: the detector sampler for
  QEC circuits. `--sampler detector|measurement` overrides it (a measurement sampler
  samples every shot, since the recorded postselection is of detectors).
- Every round times a fresh process per side, under a memory cap, with numba's
  cache warm: a process's memory layout and core placement shift its rate by
  several percent, and kernels loaded from the cache have behaved differently from
  freshly compiled ones (the cached case is what users get). A worker whose warmup
  compiled a kernel is started again.
- The first 3 rounds' processes start at once, so their imports and compiles
  overlap; each then times calls (calibrated once per device, about 0.3 s, best of
  2) when its round comes. A call returns at most about 256 MB of rows (bit-packed,
  kept rows only), so a GPU's per-call overhead does not dominate. With `--ref` the rounds stop after 3 when every round agrees on the
  verdict, else they go on to `--rounds` (5).
- `--thorough` runs each round's process from start to end on its own, every round,
  with 0.5 s calls (best of 3) and a prime run first: about 4x slower. Use it for the
  table in a pull request, and when compile time matters: without it the processes
  compile at the same time, so the compile verdict needs a 25% difference.
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
- Benchmark with the detector sampler by default. When a circuit records the
  measurement sampler, or the change touches either sampler, run both
  (`--sampler detector`, `--sampler measurement`) and report both.
- For absolute numbers (a release note, a comparison with another simulator), also
  give the all-core rate (`--threads N`), the GPU rate where there is a GPU backend
  (`--device gpu`), and the boolean-output rate (`--bools`), each labeled.
- Paste the table of a `--thorough` run in the pull request, with the CPU model
  (`lscpu | grep 'Model name'`) and the thread count.

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

- Tsim on the CPU, or with `--device gpu` on a GPU backend that the checkout's
  environment has. Comparisons with other simulators (Clifft, SymFT, Magna) are
  run from the tsim-bench repository, which pins each simulator's environment.
- To add a circuit or change a suite, change tsim-bench: put the file in its paper's
  folder, add an entry to that folder's `circuits.toml` (arXiv id, sampler, postselection
  and the reason for it, see its `BENCHMARKS.md`), run `python -m tsim_bench check`
  there, then bump `REV` in `dev/tsim_dev/corpus.py`. Until the commit is pushed, point
  the harness at the checkout with `--corpus-dir` or `TSIM_BENCH_DIR`.
