---
name: profiling-tsim
description: Finds where Tsim spends compile and sampling time with `tsim-dev profile` and `tsim-dev inspect`, and turns a profile into an optimization plan. Use when asked why a circuit is slow, where time goes, what to optimize next, or before starting performance work on Tsim.
metadata:
  internal: true
---

# Profiling Tsim

## 1. Size the circuit first

```bash
uv run tsim-dev inspect cultivation_d5          # or a path to any .stim file
```

It prints qubits, T count, detectors, `active_width` (k) and `SamplerStats` without
sampling. Cost per shot grows like 2^k. Do not sample or profile above k ≈ 20 without
a smaller `--batch-size` and a memory estimate (2^k × 16 B × batch).

## 2. Measure the speed that users get

Five settings each change a rate severalfold. The harness fixes them and prints them
on every run; say which you used, and measure the others before you conclude that
Tsim is slow:

- **Sampler.** `compile_detector_sampler` is the default for QEC circuits and Tsim's
  fast path: its direct detectors never go through a state vector. Profile and
  benchmark with the detector sampler. For a circuit whose tsim-bench manifest
  records the measurement sampler, run both (`--sampler detector`,
  `--sampler measurement`): they sample different outputs. On qv_q10 the detector
  sampler returns its one observable at 60.9M shots/s per core, the measurement
  sampler its 10 measurements at 11.6M (Apple M4 Pro).
- **Output.** The harness times bit-packed rows, Tsim's own output. `--bools`
  times `sample()`'s default boolean output, which adds a conversion: 1.6-1.8x on
  surface_d7_r7 on the CPU, and 2.3-3.2x on a GPU for circuits with hundreds of
  detectors (Metal on an M4 Pro: surface_d7_r7 3.2x, end2end_d3_dsurface15 2.8x,
  hook_inject_T_d15_pd5 2.3x; under 1.3x for circuits with few detectors). A slow
  boolean rate next to a fast bit-packed one points at the conversion, not the
  sampler.
- **Threads.** One pinned thread gives the rate per core. Tsim samples on all
  performance cores by default (`numba.get_num_threads()`); the `NOTE` line says
  how many. Quote all-core rates with `--threads N`.
- **Device.** The harness compiles with `device="cpu"`, so every side samples on
  the CPU unless you pass `--device gpu` (or `--side REF:device=gpu`). On an
  Apple-silicon Mac, Tsim samples on the GPU by default for k_max <= 11, 4-19x
  faster than all CPU cores; there `--device gpu` measures what users get. The
  `device` column says where each row sampled: under `device="auto"` a GPU falls
  back to the CPU silently (components wider than 11, tiny rotations, noise beyond
  its tables), and a row that ends in `DEVICE: cpu -> metal` compares a GPU with the
  CPU. A slow GPU row with small calls or a small `batch_size` is launch overhead;
  see "Apple-silicon Macs" in the `benchmarking-tsim` skill.
- **Compile.** `compile` includes the first use of a backend in the process (about
  0.1 s for Metal's PyObjC) but not the first call, whose kernel compilation goes
  into `warmup_s` in the JSON.

```bash
uv run tsim-dev bench qv_q10 --sampler detector               # and --sampler measurement
uv run tsim-dev bench qv_q10 --threads 10 --device gpu --bools # what a user's sample() gets
```

## 3. Profile one run

```bash
uv run tsim-dev profile cultivation_d5 --seconds 5
```

The run is a fresh process, one pinned thread, warm numba cache, under a memory cap,
with bit-packed output on the CPU (`--threads`, `--device` and `--sampler` as in
step 2). py-spy sees the host only: on a GPU the sampling phase shows the wait for
the device. For Metal's time per kernel, see "Apple-silicon Macs" in the
`benchmarking-tsim` skill (`runtime.PROFILE`).
The output splits samples by phase (`compile`, `sample`) and prints three tables:

- **innermost Tsim frame (kernel callers):** read this first. Jitted numba code has
  no symbols, so time inside a kernel is charged to the Python function that called
  it. On cultivation_d5 at 1.0.0 this shows 82% in `sample_words`
  (`noise/channels.py`) and 14% in `_run_chunk` (`statevector/shots.py`): noise
  sampling, not the state vector, is the cost there.
- **innermost frame:** separates `[jitted numba code]` from numpy, libc (memcpy,
  malloc) and Python overhead. A large non-jitted share means work outside kernels.
- **Tsim frames, inclusive:** the call tree, for compile time in particular.

The folded stacks are saved for flame graphs (`speedscope <file>` or
`flamegraph.pl`).

Profile at least two families before choosing a target; the hot spot differs
between circuits with mostly direct detectors (cultivation, surface codes), wide
quantum components (quantum volume) and many rotations (coherent noise).

## 4. Inside a kernel

The profile stops at the kernel boundary. To compare variants of a kernel, change
it and use `uv run tsim-dev bench <circuit> --ref origin/main`: that is the
measurement that counts. Micro-benchmarks of a kernel alone have misled before
(cache effects, struct passing by value), so confirm any gain end to end.

Hardware counters (`perf stat`) need `/proc/sys/kernel/perf_event_paranoid` at 1 or
lower; ask the user rather than changing it.

## 5. Turn it into a plan

- Write down the target (phase, function, share of time) and the circuits it
  affects, before changing code.
- Estimate the ceiling: removing a function that takes 14% gives at most 1.16x.
- Prefer one general change over special cases for particular circuits, and drop
  ideas whose gain is small compared with the code they add.
- After the change, profile again to confirm the share moved, then follow the
  `benchmarking-tsim` skill for the numbers that go in the pull request.
