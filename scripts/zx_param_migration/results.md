# zx_param migration: comparison with today's Tsim

- **Old:** today's Tsim, main `3dacc1a` (built on pyzx-param).
- **New:** the migrated Tsim, branch `zx-param-migration` (built on zx_param).

Each version ran `probabilities.py`, `rates.py` and `benchmark.py` from
`scripts/zx_param_migration/` in its own checkout and environment, on the JAX
CPU backend. This file is written by `compare.py`.

Noisy circuits are not compared probability by probability: `ChannelSampler`
simplifies the noise channels with the error-basis transform, so seeded error
draws can differ between the versions. Exact probabilities are compared on
noiseless circuits, and noisy circuits through their detector and observable
rates.

## Verdict

**PASS.** All 8 exact-probability rows and all 7 detector-rate rows agree within tolerance. Benchmarks are reported below, not gated. Every circuit ran in both versions; none was dropped.

## Exact probabilities

Circuits: `SMALL`, `generated()` and `coherent_d3_r1`, each without noise.
States: 8 measurement records drawn by the migrated Tsim
(`compile_sampler(seed=0).sample(8, batch_size=8)`) plus the all-zeros record.
Each version computes
`CompiledStateProbs(circuit, strategy="cat5", seed=0).probability_of(state, batch_size=1)`
(joint mode). A row passes if
`np.allclose(p_new, p_old, rtol=0.0001, atol=1e-07)`, that is
`|p_new - p_old| <= 1e-07 + 0.0001 * |p_old|` for every state. The relative
deviation is taken over states with `p_old > 0`.

| Circuit | States | Max abs deviation | Max rel deviation | Result |
|---|---:|---:|---:|---|
| bell_t | 9 | 0.00e+00 | 0.00e+00 | pass |
| u3_rotations | 9 | 5.96e-08 | 1.24e-07 | pass |
| feedforward | 9 | 0.00e+00 | 0.00e+00 | pass |
| mpp_and_resets | 9 | 0.00e+00 | 0.00e+00 | pass |
| toffoli | 9 | 0.00e+00 | 0.00e+00 | pass |
| surface_d3_r2 | 9 | 0.00e+00 | 0.00e+00 | pass |
| repetition_d5_r3 | 9 | 0.00e+00 | 0.00e+00 | pass |
| coherent_d3_r1 | 9 | 0.00e+00 | 0.00e+00 | pass |

## Detector and observable rates

Circuits: `generated()` (cat5) and `clifft()` (each with its strategy), with
noise, in detector mode. Each version samples
`compile_detector_sampler(strategy=..., seed=0).sample(shots, batch_size=b,
append_observables=True)` with 200,000 shots (20,000 for distillation and
msc_d3) and b = 8,192 (64 for msc_d3), and takes the mean of every column. A
column passes if `|p_old - p_new| <= 5 * sigma + 0.0001`, where
`sigma = sqrt(p_old * (1 - p_old) / n_old + p_new * (1 - p_new) / n_new)`; a row
passes if every column does. The z-score of a column is
`|p_old - p_new| / sigma`.

| Circuit | Columns | Shots (old / new) | Max abs deviation | Worst z | Result |
|---|---:|---:|---:|---:|---|
| surface_d3_r2 | 17 | 200,000 / 200,000 | 0.00e+00 | 0.00 | pass |
| repetition_d5_r3 | 17 | 200,000 / 200,000 | 0.00e+00 | 0.00 | pass |
| surface_d7_r7 | 337 | 200,000 / 200,000 | 0.00e+00 | 0.00 | pass |
| coherent_d3_r1 | 9 | 200,000 / 200,000 | 0.00e+00 | 0.00 | pass |
| coherent_d5_r1 | 25 | 200,000 / 200,000 | 0.00e+00 | 0.00 | pass |
| distillation | 45 | 20,000 / 20,000 | 8.75e-03 | 2.40 | pass |
| msc_d3 | 21 | 20,000 / 20,000 | 4.35e-03 | 1.87 | pass |

## Benchmarks

Circuits: `clifft()` with noise, detector mode. Compile time is one
`compile_detector_sampler(strategy=..., seed=0)` call. Terms is the sum of
`num_graphs` over every compiled scalar graph of every component (0 when all
outputs take the direct path). After one warm-up `sample(4096)` call, one
`sample(65536, batch_size=8192)` call is timed (`sample(4096, batch_size=64)`
for msc_d3). Ratios are new/old: above 1 means the new compile is slower, and
below 1 means the new sampling is slower. Single runs, one version after the
other, on a shared machine (the load column is the 1-minute load average after
each circuit, old / new). Unlike the clifft-bench harness, no postselection
mask is passed, so absolute rates differ from that harness's. Benchmarks are
reported, not gated.

| Circuit | Strategy | Compile s (old / new) | New/old | Terms (old / new) | Shots/s (old / new) | New/old | Load (old / new) |
|---|---|---:|---:|---:|---:|---:|---:|
| surface_d7_r7 | cat5 | 2.44 / 0.922 | 0.38 | 0 / 0 | 1,331,399 / 1,315,402 | 0.99 | 8.0 / 13.1 |
| coherent_d3_r1 | cat5 | 0.0229 / 0.018 | 0.78 | 0 / 0 | 265,238,782 / 274,448,010 | 1.03\* | 8.0 / 13.1 |
| coherent_d5_r1 | cat5 | 0.0521 / 0.043 | 0.83 | 0 / 0 | 75,855,536 / 94,506,082 | 1.25\* | 8.0 / 13.1 |
| distillation | cat5 | 1.11 / 0.678 | 0.61 | 520 / 446 | 94,414 / 164,872 | 1.75 | 8.0 / 13.1 |
| msc_d3 | cutting | 7.31 / 7.64 | 1.05 | 3,873 / 3,873 | 373 / 1,771 | 4.75 | 14.0 / 10.8 |

\* A timed call took under 10 ms, so this ratio is noisy.

Tsim imported by the benchmark runs: old `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/baseline-3dacc1a/src/tsim/__init__.py`, new `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/zx-param-migration/src/tsim/__init__.py`.
