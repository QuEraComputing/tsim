# strategy="statevector" on the clifft-bench circuits

- **Date:** 2026-10-10
- **Machine:** Apple M4 Pro, 14 cores (10 performance, 4 efficiency)
- **Backend:** JAX CPU, default settings (no `XLA_FLAGS` or thread limits), so
  XLA may use every core
- **Threads used:** the full run took 219.23 s user + 14.67 s system over
  82.98 s wall (shell `time`), about 2.8 cores busy on average. Tsim's shots/s
  below are multi-threaded, not one-core figures.
- **JAX:** 0.9.2
- **Commit:** `bf80f7e`

Command, from the repository root:

    uv run python scripts/statevector/benchmark.py --seconds 10

See the docstring of `benchmark.py` for what is timed and counted. Amplitudes
are complex64 (the default). "warmup s" is the first sample call, which
includes JIT compilation and one full sample call of the workload's shots per
call (see JIT time below).

## Measured

| circuit | compile s | warmup s | shots/s | acceptance | Clifft acceptance | check | logical errors / accepted | sampler |
|---|---:|---:|---:|---:|---:|---|---:|---|
| distillation | 0.16 | 0.61 | 2,926,016 | 0.0799 | 0.0798 | ok | 990898 / 2344917 | `CompiledDetectorSampler(40 direct, 1 statevector components (width ≤ 3, 15 actions), 0.0 graphs, 473 error channel bits, 5 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 579 B)` |
| cultivation_d3 | 0.11 | 0.74 | 1,189,716 | 0.6866 | 0.6868 | ok | 6 / 8279256 | `CompiledDetectorSampler(16 direct, 1 statevector components (width ≤ 5, 36 actions), 0.0 graphs, 268 error channel bits, 5 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 1.6 kB)` |
| coherent_d3_r3 | 0.07 | 1.02 | 570,139 | 0.3732 | 0.3732 | ok | 0 / 2152025 | `CompiledDetectorSampler(17 direct, 1 statevector components (width ≤ 5, 73 actions), 0.0 graphs, 54 error channel bits, 8 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 3.0 kB)` |
| cultivation_d5 | 0.92 | 0.69 | 594,757 | 0.1442 | 0.1440 | ok | 0 / 859817 | `CompiledDetectorSampler(93 direct, 1 statevector components (width ≤ 6, 109 actions), 0.0 graphs, 1688 error channel bits, 15 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 6.7 kB)` |
| coherent_d5_r5 | 0.74 | 18.72 | 901 | 0.0094 | 0.0078 | ok | 0 / 154 | `CompiledDetectorSampler(73 direct, 1 statevector components (width ≤ 12, 427 actions), 0.0 graphs, 234 error channel bits, 48 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 26.0 kB)` |

## Reference

| circuit | Clifft 0.10.0 shots/s (one core, AMD EPYC 9R14) | Tsim stabilizer rank on this machine |
|---|---:|---:|
| distillation | 1.87M | 603k (cat5) |
| cultivation_d3 | 2.82M | 567 (cutting) |
| coherent_d3_r3 | 1.05M | does not compile in 120 s |
| cultivation_d5 | 197k | does not compile in 120 s |
| coherent_d5_r5 | 5.9k | does not compile in 120 s |

## Logical-error rate

Tsim's rate is logical errors / accepted from the measured table. Clifft
0.10.0's rates are from the design spec (section 11). "Expected" is accepted
times Clifft's rate.

| circuit | Tsim errors / accepted | Tsim rate | Clifft 0.10.0 rate | expected errors at Clifft's rate |
|---|---:|---:|---:|---:|
| distillation | 990898 / 2344917 | 0.4226 | 0.422 | 989,555 (binomial sd 756) |
| cultivation_d3 | 6 / 8279256 | 7.2e-7 | 9.6e-7 | 7.95 |
| coherent_d3_r3 | 0 / 2152025 | 0 | 5.1e-8 | 0.11 |
| cultivation_d5 | 0 / 859817 | 0 | 0 (in 3.0e7 shots) | 0 |
| coherent_d5_r5 | 0 / 154 | 0 | 0 (in 8.8e5 shots) | 0 |

distillation is 1,343 errors (1.8 sd) above the count expected at 0.422. The
reference's third-digit rounding (±0.0005) by itself moves that expected count
by up to ±1,172. For cultivation_d3, a Poisson count with mean 7.95 is 6 or
fewer with probability 0.32. For coherent_d3_r3, a Poisson count with mean
0.11 is 0 with probability 0.90.

## JIT time

The script does not time JIT separately. Estimate it as warmup s minus one
call's time, where one call's time is shots per call divided by the measured
shots/s. This assumes the warmup call samples at the timed rate. The estimate
also includes any other first-call overhead, and warmup s is rounded to
0.01 s.

| circuit | warmup s | shots per call | one call s | JIT estimate s |
|---|---:|---:|---:|---:|
| distillation | 0.61 | 262,144 | 0.090 | 0.52 |
| cultivation_d3 | 0.74 | 262,144 | 0.220 | 0.52 |
| coherent_d3_r3 | 1.02 | 262,144 | 0.460 | 0.56 |
| cultivation_d5 | 0.69 | 65,536 | 0.110 | 0.58 |
| coherent_d5_r5 | 18.72 | 16,384 | 18.184 | 0.54 |

## Summary

All five acceptances agree with Clifft 0.10.0 within 5 standard errors, and
all five logical-error counts are consistent with Clifft's rates. Every
circuit compiles in under 1 s, including the three that the stabilizer rank
strategy does not compile in 120 s, and JIT takes about 0.5 s. Tsim's
multi-threaded shots/s on this machine (about 2.8 cores busy on average) is
above Clifft's one-core EPYC figure on distillation and cultivation_d5, and
below it on cultivation_d3, coherent_d3_r3 and coherent_d5_r5 (901 against
5.9k). This compares different machines and thread counts, so it is not a
per-core comparison.

Run notes:

- coherent_d5_r5: one sample call of 16,384 shots takes about 18 s, so the
  timed window holds a single call and its acceptance check rests on 16,384
  attempted shots (154 accepted).
- coherent_d5_r5 printed six `UserWarning`s from `sampler.py` ("A marginal
  probability was not normalized correctly"), with deviations from 2.0e-05 to
  4.1e-05 (the warning threshold is 1e-05). With `JAX_ENABLE_X64=1`
  (complex128), `--seconds 1 coherent_d5_r5` printed no warning, with
  acceptance 0.0079 (129 accepted of 16,384) at 599 shots/s.
