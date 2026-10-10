# strategy="statevector" on the clifft-bench circuits

- **Date:** 2026-10-10
- **Machine:** Apple M4 Pro, 14 cores (10 performance, 4 efficiency)
- **Backend:** JAX CPU, default settings (no `XLA_FLAGS` or thread limits), so
  XLA may use every core
- **Threads used:** the full run took 227.61 s user + 14.68 s system over
  84.06 s wall (shell `time`), about 2.9 cores busy on average. Tsim's shots/s
  below are multi-threaded, not one-core figures.
- **JAX:** 0.9.2
- **Commit:** `0d6b4ab`, which includes the executor fixes of the final review:
  outcomes drawn with 2^-64 resolution, outcome probabilities computed from
  the outcome weights, and dtype-aware norm warnings

Command, from the repository root:

    uv run python scripts/statevector/benchmark.py --seconds 10

See the docstring of `benchmark.py` for what is timed and counted. Amplitudes
are complex64 (the default). "warmup s" is the first sample call, which
includes JIT compilation and one full sample call of the workload's shots per
call (see JIT time below). The script sets every batch size explicitly, so the
automatic batch-size cap (2^16 lanes) does not apply here.

## Measured

| circuit | compile s | warmup s | shots/s | acceptance | Clifft acceptance | check | logical errors / accepted | sampler |
|---|---:|---:|---:|---:|---:|---|---:|---|
| distillation | 0.17 | 0.63 | 2,869,787 | 0.0799 | 0.0798 | ok | 973000 / 2302973 | `CompiledDetectorSampler(40 direct, 1 statevector components (width ≤ 3, 15 actions), 0.0 graphs, 473 error channel bits, 5 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 579 B)` |
| cultivation_d3 | 0.11 | 0.78 | 1,096,444 | 0.6866 | 0.6868 | ok | 4 / 7559234 | `CompiledDetectorSampler(16 direct, 1 statevector components (width ≤ 5, 36 actions), 0.0 graphs, 268 error channel bits, 5 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 1.6 kB)` |
| coherent_d3_r3 | 0.07 | 1.07 | 522,613 | 0.3730 | 0.3732 | ok | 0 / 1955689 | `CompiledDetectorSampler(17 direct, 1 statevector components (width ≤ 5, 73 actions), 0.0 graphs, 54 error channel bits, 8 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 3.0 kB)` |
| cultivation_d5 | 0.92 | 0.69 | 572,691 | 0.1441 | 0.1440 | ok | 0 / 831104 | `CompiledDetectorSampler(93 direct, 1 statevector components (width ≤ 6, 109 actions), 0.0 graphs, 1688 error channel bits, 15 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 6.7 kB)` |
| coherent_d5_r5 | 0.73 | 19.22 | 874 | 0.0094 | 0.0078 | ok | 0 / 154 | `CompiledDetectorSampler(73 direct, 1 statevector components (width ≤ 12, 427 actions), 0.0 graphs, 234 error channel bits, 48 outputs for largest cc, ≤ 0 parameters, 0.0 A terms, 0.0 B terms, 0.0 C terms, 0.0 D terms, 0.0 E terms, 26.0 kB)` |

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
times Clifft's rate. A count of 0 is written as a 95% upper bound, 3/n for n
shots rounded up (n accepted for Tsim; n as quoted by the spec for Clifft).

| circuit | Tsim errors / accepted | Tsim rate | Clifft 0.10.0 rate | expected errors at Clifft's rate |
|---|---:|---:|---:|---:|
| distillation | 973000 / 2302973 | 0.4225 | 0.422 | 971,855 (binomial sd 750) |
| cultivation_d3 | 4 / 7559234 | 5.3e-7 | 9.6e-7 | 7.26 |
| coherent_d3_r3 | 0 / 1955689 | < 1.6e-6 | 5.1e-8 | 0.10 |
| cultivation_d5 | 0 / 831104 | < 3.7e-6 | < 1.0e-7 (0 in 3.0e7 shots) | < 0.09 |
| coherent_d5_r5 | 0 / 154 | < 2.0e-2 | < 3.5e-6 (0 in 8.8e5 shots) | < 0.001 |

distillation is 1,145 errors (1.5 sd) above the count expected at 0.422. The
reference's third-digit rounding (±0.0005) by itself moves that expected count
by up to ±1,151. For cultivation_d3, a Poisson count with mean 7.26 is 4 or
fewer with probability 0.15. For coherent_d3_r3, a Poisson count with mean
0.10 is 0 with probability 0.90.

The previous version of this section (commit `bf80f7e`) could not detect the
outcome-sampling floor that the final review found: complex64 sampling gave
an impossible outcome with probability ~1.2e-7 per measurement, which on
cultivation_d5 flipped observable 0 in about 1.0e-7 of the accepted shots.
Over the 859,817 accepted shots of that run this predicts 0.09 extra logical
errors, and the acceptance shift is far below the 5-sd check, so neither
column could show it. Resolving a floor of that size needs ~3e7 or more
accepted shots. The run above includes the fix; its counts are again
consistent with Clifft's rates, at the same limited resolution.

## JIT time

The script does not time JIT separately. Estimate it as warmup s minus one
call's time, where one call's time is shots per call divided by the measured
shots/s. This assumes the warmup call samples at the timed rate. The estimate
also includes any other first-call overhead, and warmup s is rounded to
0.01 s.

| circuit | warmup s | shots per call | one call s | JIT estimate s |
|---|---:|---:|---:|---:|
| distillation | 0.63 | 262,144 | 0.091 | 0.54 |
| cultivation_d3 | 0.78 | 262,144 | 0.239 | 0.54 |
| coherent_d3_r3 | 1.07 | 262,144 | 0.502 | 0.57 |
| cultivation_d5 | 0.69 | 65,536 | 0.114 | 0.58 |
| coherent_d5_r5 | 19.22 | 16,384 | 18.746 | 0.47 |

## Summary

All five acceptances agree with Clifft 0.10.0 within 5 standard errors, and
all five logical-error counts are consistent with Clifft's rates. Every
circuit compiles in under 1 s, including the three that the stabilizer rank
strategy does not compile in 120 s, and JIT takes about 0.5 s. Tsim's
multi-threaded shots/s on this machine (about 2.9 cores busy on average) is
above Clifft's one-core EPYC figure on distillation and cultivation_d5, and
below it on cultivation_d3, coherent_d3_r3 and coherent_d5_r5 (874 against
5.9k). This compares different machines and thread counts, so it is not a
per-core comparison. Against the previous run (commit `bf80f7e`), shots/s is
2% to 8% lower (distillation 2.93M to 2.87M, cultivation_d3 1.19M to 1.10M,
coherent_d3_r3 570k to 523k, cultivation_d5 595k to 573k, coherent_d5_r5 901
to 874); the fixes draw two uint32 words per outcome and sum two outcome
weights per measurement.

Run notes:

- coherent_d5_r5: one sample call of 16,384 shots takes about 19 s, so the
  timed window holds a single call and its acceptance check rests on 16,384
  attempted shots (154 accepted).
- No norm warning was printed. The previous run printed six on
  coherent_d5_r5, with deviations from 2.0e-05 to 4.1e-05 against the old
  1e-05 threshold. With the fixed kernels, four batches of 4,096 lanes of
  coherent_d5_r5 reach a largest norm deviation of 1.4e-06 to 1.6e-06, and
  complex64 components now warn above 1e-03.
