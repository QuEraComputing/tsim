# strategy="active_statevector" on the clifft-bench circuits

- **Date:** 2026-10-10
- **Machine:** Apple M4 Pro, 14 cores (10 performance, 4 efficiency)
- **Tsim:** commit `883b125` (the sampler code is unchanged since `0d6b4ab`),
  JAX 0.9.2 on CPU, complex64 amplitudes (the default)
- **Clifft:** 0.11.0 from PyPI, in its own environment, on the same machine
- **Threads:** "all threads" is each library's default: XLA's CPU thread pools
  for Tsim, `threads="auto"` for Clifft. "1 thread" is `NPROC=1` for Tsim (XLA
  sizes its CPU thread pools from `NPROC`; macOS has no CPU affinity) and
  `threads=1` for Clifft. "cores busy" is the process CPU time divided by wall
  time over the timed window; it is 1.0 in every 1-thread row.

Commands, from the repository root (`benchmark_clifft.py` describes how to
install Clifft):

    uv run python scripts/statevector/benchmark.py --seconds 10
    NPROC=1 uv run python scripts/statevector/benchmark.py --seconds 10
    /tmp/clifft-venv/bin/python scripts/statevector/benchmark_clifft.py --seconds 10

See the docstrings of both scripts for what is timed and counted. Both
postselect on every detector, as clifft-bench does. Tsim's "warmup s" is the
first sample call, which includes JIT compilation and one full sample call of
the workload's shots per call (see JIT time below). The scripts set every batch
size explicitly, so Tsim's automatic batch-size cap (2^16 lanes) does not apply.

## Same-machine comparison

Shots/s from the measured tables below. Tsim's width is the width of its state
vector; Clifft's k_max is its peak active width under its default passes.

| circuit | Tsim width | Clifft k_max | Tsim, 1 thread | Clifft, 1 thread | Tsim, all threads | Clifft, all threads |
|---|---:|---:|---:|---:|---:|---:|
| distillation | 3 | 5 | 2,462,770 | 2,755,248 | 2,869,491 | 20,196,109 |
| cultivation_d3 | 5 | 4 | 562,902 | 4,742,046 | 1,108,828 | 30,405,370 |
| coherent_d3_r3 | 5 | 5 | 248,544 | 1,920,603 | 527,961 | 13,664,724 |
| cultivation_d5 | 6 | 10 | 312,294 | 136,869 | 572,464 | 1,207,485 |
| coherent_d5_r5 | 12 | 13 | 272 | 3,692 | 886 | 25,476 |

Clifft's shots/s divided by Tsim's, and the cores each keeps busy on all
threads:

| circuit | 1 thread | all threads | cores busy, Tsim / Clifft |
|---|---:|---:|---:|
| distillation | 1.1 | 7.0 | 1.3 / 8.7 |
| cultivation_d3 | 8.4 | 27 | 2.5 / 8.4 |
| coherent_d3_r3 | 7.7 | 26 | 2.9 / 9.1 |
| cultivation_d5 | 0.44 | 2.1 | 2.4 / 10.7 |
| coherent_d5_r5 | 13.6 | 29 | 3.7 / 9.5 |

## Measured

### Tsim, all threads

| circuit | compile s | warmup s | shots/s | cores busy | acceptance | Clifft acceptance | check | logical errors / accepted | sampler |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| distillation | 0.17 | 0.63 | 2,869,491 | 1.3 | 0.0799 | 0.0798 | ok | 973000 / 2302973 | `CompiledDetectorSampler(40 direct, 1 statevector components (width ≤ 3, 15 actions), 0.0 graphs, 473 error channel bits, 5 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 579 B)` |
| cultivation_d3 | 0.11 | 0.76 | 1,108,828 | 2.5 | 0.6866 | 0.6868 | ok | 4 / 7739165 | `CompiledDetectorSampler(16 direct, 1 statevector components (width ≤ 5, 36 actions), 0.0 graphs, 268 error channel bits, 5 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 1.6 kB)` |
| coherent_d3_r3 | 0.07 | 1.05 | 527,961 | 2.9 | 0.3731 | 0.3732 | ok | 0 / 2053991 | `CompiledDetectorSampler(17 direct, 1 statevector components (width ≤ 5, 73 actions), 0.0 graphs, 54 error channel bits, 8 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 3.0 kB)` |
| cultivation_d5 | 0.92 | 0.66 | 572,464 | 2.4 | 0.1441 | 0.1440 | ok | 0 / 831104 | `CompiledDetectorSampler(93 direct, 1 statevector components (width ≤ 6, 109 actions), 0.0 graphs, 1688 error channel bits, 15 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 6.7 kB)` |
| coherent_d5_r5 | 0.73 | 19.32 | 886 | 3.7 | 0.0094 | 0.0078 | ok | 0 / 154 | `CompiledDetectorSampler(73 direct, 1 statevector components (width ≤ 12, 427 actions), 0.0 graphs, 234 error channel bits, 48 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 26.0 kB)` |

### Tsim, 1 thread (`NPROC=1`)

| circuit | compile s | warmup s | shots/s | cores busy | acceptance | Clifft acceptance | check | logical errors / accepted | sampler |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| distillation | 0.17 | 0.64 | 2,462,770 | 1.0 | 0.0798 | 0.0798 | ok | 831265 / 1967388 | `CompiledDetectorSampler(40 direct, 1 statevector components (width ≤ 3, 15 actions), 0.0 graphs, 473 error channel bits, 5 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 579 B)` |
| cultivation_d3 | 0.12 | 1.00 | 562,902 | 1.0 | 0.6867 | 0.6868 | ok | 0 / 3960347 | `CompiledDetectorSampler(16 direct, 1 statevector components (width ≤ 5, 36 actions), 0.0 graphs, 268 error channel bits, 5 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 1.6 kB)` |
| coherent_d3_r3 | 0.07 | 1.57 | 248,544 | 1.0 | 0.3733 | 0.3732 | ok | 0 / 978504 | `CompiledDetectorSampler(17 direct, 1 statevector components (width ≤ 5, 73 actions), 0.0 graphs, 54 error channel bits, 8 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 3.0 kB)` |
| cultivation_d5 | 0.92 | 0.75 | 312,294 | 1.0 | 0.1440 | 0.1440 | ok | 0 / 453106 | `CompiledDetectorSampler(93 direct, 1 statevector components (width ≤ 6, 109 actions), 0.0 graphs, 1688 error channel bits, 15 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 6.7 kB)` |
| coherent_d5_r5 | 0.75 | 58.14 | 272 | 1.0 | 0.0094 | 0.0078 | ok | 0 / 154 | `CompiledDetectorSampler(73 direct, 1 statevector components (width ≤ 12, 427 actions), 0.0 graphs, 234 error channel bits, 48 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 26.0 kB)` |

### Clifft 0.11.0

"batch" is the faster of scalar execution (1) and packed lanes of 2048 shots.
Clifft rejects packed lanes of 2048 on coherent_d5_r5 (they exceed its 64 MiB
packed-state limit at k_max 13), so those rows are scalar.

| circuit | threads | batch | compile s | k_max | shots/s | cores busy | acceptance | Clifft 0.10.0 acceptance | check | logical errors / accepted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| distillation | 1 | 2048 | 0.00 | 5 | 2,755,248 | 1.0 | 0.0798 | 0.0798 | ok | 935589 / 2217580 |
| distillation | auto | 2048 | 0.00 | 5 | 20,196,109 | 8.7 | 0.0797 | 0.0798 | ok | 6798775 / 16115015 |
| cultivation_d3 | 1 | 2048 | 0.00 | 4 | 4,742,046 | 1.0 | 0.6868 | 0.6868 | ok | 42 / 32589642 |
| cultivation_d3 | auto | 2048 | 0.00 | 4 | 30,405,370 | 8.4 | 0.6868 | 0.6868 | ok | 222 / 208838618 |
| coherent_d3_r3 | 1 | 2048 | 0.00 | 5 | 1,920,603 | 1.0 | 0.3732 | 0.3732 | ok | 0 / 7239813 |
| coherent_d3_r3 | auto | 2048 | 0.00 | 5 | 13,664,724 | 9.1 | 0.3732 | 0.3732 | ok | 1 / 51070460 |
| cultivation_d5 | 1 | 1 | 0.01 | 10 | 136,869 | 1.0 | 0.1444 | 0.1440 | ok | 0 / 198758 |
| cultivation_d5 | auto | 1 | 0.01 | 10 | 1,207,485 | 10.7 | 0.1440 | 0.1440 | ok | 0 / 1746413 |
| coherent_d5_r5 | 1 | 1 | 0.01 | 13 | 3,692 | 1.0 | 0.0075 | 0.0078 | ok | 0 / 369 |
| coherent_d5_r5 | auto | 1 | 0.01 | 13 | 25,476 | 9.5 | 0.0074 | 0.0078 | ok | 0 / 1931 |

## Active width

| circuit | Clifft, no passes | Clifft, default passes | Clifft + `ActiveWidthSchedulePass` | Tsim, detectors | Tsim, measurements |
|---|---:|---:|---:|---:|---:|
| distillation | 5 | 5 | 3 | 3 | 3 |
| cultivation_d3 | 4 | 4 | 4 | 5 | 5 |
| coherent_d3_r3 | 8 | 5 | 4 | 5 | 4 |
| cultivation_d5 | 10 | 10 | 10 | 6 | 6 |
| coherent_d5_r5 | 24 | 13 | 13 | 12 | 15 |

Clifft's columns are `peak_active_width` of `clifft.compile(text,
hir_passes=None)`, of `clifft.compile(text)`, and of `clifft.compile` with
`default_hir_pass_manager()` plus `ActiveWidthSchedulePass()`, which is opt-in
in 0.11.0. The benchmark uses the default passes. The Clifft paper's table
matches the no-pass column. Tsim's columns are the largest `width` over the
statevector components of `compile_detector_sampler` (the benchmark) and of
`compile_sampler`.

## Reference

| circuit | Clifft 0.10.0 shots/s (one core, AMD EPYC 9R14) | Tsim stabilizer rank on this machine |
|---|---:|---:|
| distillation | 1.87M | 603k (cat5) |
| cultivation_d3 | 2.82M | 567 (cutting) |
| coherent_d3_r3 | 1.05M | does not compile in 120 s |
| cultivation_d5 | 197k | does not compile in 120 s |
| coherent_d5_r5 | 5.9k | does not compile in 120 s |

Clifft 0.10.0's figures are from clifft-bench release-v1-20260909-130000, on a
different machine.

## Logical-error rate

Tsim's counts pool the benchmark's run (seed 0) with independent runs:

    uv run python scripts/statevector/benchmark.py --seconds 120 --seed 1 cultivation_d3
    uv run python scripts/statevector/benchmark.py --seconds 45 --seed 1 distillation
    uv run python scripts/statevector/benchmark.py --seconds 45 --seed 2 distillation

which gave 78 / 89470166, 4242388 / 10057137 and 4275565 / 10138745 (errors /
accepted). The references are Clifft 0.11.0 on this machine (both thread
settings pooled), and Clifft 0.10.0 and SymFT from the raw data of clifft-bench
release-v1-20260909-130000. z is Tsim's deviation from each reference in
standard deviations, counting the statistical error of both. A count of 0 is
written as a 95% upper bound, 3/n.

| circuit | Tsim errors / accepted | Tsim rate | Clifft 0.11.0 | Clifft 0.10.0 | SymFT |
|---|---:|---:|---:|---:|---:|
| distillation | 9490953 / 22498855 | 0.42184 | 0.42189 (z −0.3) | 0.42152 (z +2.2) | 0.42158 (z +1.8) |
| cultivation_d3 | 82 / 97209331 | 8.4e-7 | 1.09e-6 (z −2.0) | 9.6e-7 (z −1.0) | 8.5e-7 (z −0.1) |
| coherent_d3_r3 | 0 / 2053991 | < 1.5e-6 | 1 / 58310273 | 3 / 58747248 | 1 / 31579774 |
| cultivation_d5 | 0 / 831104 | < 3.6e-6 | 0 / 1945171 | 0 / 4272772 | 0 / 3685293 |
| coherent_d5_r5 | 0 / 154 | < 2.0e-2 | 0 / 2300 | 0 / 6851 | 0 / 63 |

The references scatter by as much among themselves: Clifft 0.11.0 is 2.4 sd
from Clifft 0.10.0 on distillation and 2.8 sd from SymFT on cultivation_d3, and
Tsim lies within that scatter. For coherent_d3_r3, the pooled rate of Clifft
0.10.0 and 0.11.0 (4 / 117057521) predicts 0.07 Tsim errors.

The outcome-sampling floor that the final review found (complex64 drew an
impossible outcome with probability ~1.2e-7 per measurement) was fixed in
`0d6b4ab`, before all of these runs.

## JIT time

The script does not time JIT separately. Estimate it from the all-threads run
as warmup s minus one call's time, where one call's time is shots per call
divided by the measured shots/s. This assumes the warmup call samples at the
timed rate. The estimate also includes any other first-call overhead, and
warmup s is rounded to 0.01 s.

| circuit | warmup s | shots per call | one call s | JIT estimate s |
|---|---:|---:|---:|---:|
| distillation | 0.63 | 262,144 | 0.091 | 0.54 |
| cultivation_d3 | 0.76 | 262,144 | 0.236 | 0.52 |
| coherent_d3_r3 | 1.05 | 262,144 | 0.497 | 0.55 |
| cultivation_d5 | 0.66 | 65,536 | 0.114 | 0.55 |
| coherent_d5_r5 | 19.32 | 16,384 | 18.492 | 0.83 |

## Summary

All acceptances, Tsim's on both thread settings and Clifft 0.11.0's, agree with
Clifft 0.10.0's within 5 standard errors, and Tsim's logical-error rates agree
with the exact references within their scatter. On this machine Clifft 0.11.0
is faster than Tsim on four of the five circuits. On one thread it is 1.1×
faster on distillation, 7.7× and 8.4× on coherent_d3_r3 and cultivation_d3, and
13.6× on coherent_d5_r5. Tsim is faster only on cultivation_d5, by 2.3× on one
thread, where its width is 6 against Clifft's 10. On all threads the gaps grow
(2.1× on cultivation_d5, 7× to 29× on the others), because Clifft keeps 8 to 11
cores busy and Tsim 1.3 to 3.7. Tsim compiles in 0.07 to 0.92 s, Clifft in
0.01 s or less.

The previous version of this file compared Tsim on all threads with Clifft
0.10.0's one-core EPYC figures and found Tsim faster on distillation and
cultivation_d5. Measured on the same machine against Clifft 0.11.0, that holds
only for cultivation_d5, and only per thread.

Run notes:

- coherent_d5_r5: one Tsim sample call of 16,384 shots takes about 18 s on all
  threads and 60 s on one, so each timed window holds a single call and the
  acceptance check rests on 16,384 attempted shots (154 accepted).
- Clifft's coherent_d5_r5 rows come from a rerun of that workload, after the
  script learned to skip a batch size that Clifft rejects; the other rows are
  unaffected by that change.
- No norm warning was printed.
- Whole runs (shell `time`): Tsim on all threads took 84.4 s wall, 226.6 s user
  and 15.1 s system; Tsim on one thread took 166.2 s wall, 161.4 s user and
  4.5 s system.
