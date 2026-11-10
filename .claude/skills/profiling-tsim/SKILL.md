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

## 2. Profile one run

```bash
uv run tsim-dev profile cultivation_d5 --seconds 5
```

The run is a fresh process, one pinned thread, warm numba cache, under a memory cap.
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

## 3. Inside a kernel

The profile stops at the kernel boundary. To compare variants of a kernel, change
it and use `uv run tsim-dev bench <circuit> --ref origin/main`: that is the
measurement that counts. Micro-benchmarks of a kernel alone have misled before
(cache effects, struct passing by value), so confirm any gain end to end.

Hardware counters (`perf stat`) need `/proc/sys/kernel/perf_event_paranoid` at 1 or
lower; ask the user rather than changing it.

## 4. Turn it into a plan

- Write down the target (phase, function, share of time) and the circuits it
  affects, before changing code.
- Estimate the ceiling: removing a function that takes 14% gives at most 1.16x.
- Prefer one general change over special cases for particular circuits, and drop
  ideas whose gain is small compared with the code they add.
- After the change, profile again to confirm the share moved, then follow the
  `benchmarking-tsim` skill for the numbers that go in the pull request.
