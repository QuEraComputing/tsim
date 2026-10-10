# zx_param migration: comparison with today's Tsim

- **Old:** today's Tsim, main `3dacc1a` (built on pyzx-param).
- **New:** the migrated Tsim, branch `zx-param-migration` (built on zx_param).

Each version ran `probabilities.py`, `rates.py` and `benchmark.py` from
`scripts/zx_param_migration/` in its own checkout and environment, on the JAX
CPU backend. This file is written by `compare.py`; the Tsim each run imported is
listed at the end.

Noisy circuits are not compared probability by probability: `ChannelSampler`
simplifies the noise channels with the error-basis transform, so seeded error
draws can differ between the versions. Exact probabilities are compared on
noiseless circuits, and noisy circuits through their detector and observable
rates.

## Verdict

**PASS.** All 8 exact-probability rows and all 7 detector-rate rows agree within tolerance.

Benchmarks (3 runs, reported, not gated): worse than today: msc_d3 compile time, worse in every run (new/old by run: 1.03, 1.01, 1.08). Every other counted ratio is the same as or better than today's.

Every circuit in `circuits.py` ran in both versions; none was dropped. Three clifft-bench workloads that exceed its compile limit were never part of the comparison (see Benchmarks).

## Exact probabilities

Circuits: `SMALL`, `generated()` and `coherent_d3_r1`, each without noise.
States (written by `states.py` with the migrated Tsim, deduplicated): for
`SMALL`, all 2^n measurement records; for the others, 8 records drawn with
`compile_sampler(seed=0).sample(8, batch_size=8)`, the all-zeros record, and
the first 3 drawn records with one bit flipped in the support of detector 0, 1
and 2 respectively. Each version computes
`CompiledStateProbs(circuit, strategy="cat5", seed=0).probability_of(state, batch_size=1)`
(joint mode). A row passes if
`np.allclose(p_new, p_old, rtol=0.0001, atol=1e-07)`, that is
`|p_new - p_old| <= 1e-07 + 0.0001 * |p_old|` for every state. "Values"
counts the distinct probabilities among the states (values closer than that
tolerance count once), and "Zero" counts the states with `p_old <= 1e-07`.
The relative deviation is taken over states with `p_old > 0`.

| Circuit | States | Values | Zero | Max abs deviation | Max rel deviation | Result |
|---|---:|---:|---:|---:|---:|---|
| bell_t | 4 | 1 | 0 | 0.00e+00 | 0.00e+00 | pass |
| u3_rotations | 4 | 4 | 0 | 5.96e-08 | 1.85e-07 | pass |
| feedforward | 4 | 2 | 0 | 0.00e+00 | 0.00e+00 | pass |
| mpp_and_resets | 64 | 3 | 56 | 0.00e+00 | 0.00e+00 | pass |
| toffoli | 8 | 1 | 0 | 0.00e+00 | 0.00e+00 | pass |
| surface_d3_r2 | 12 | 2 | 3 | 0.00e+00 | 0.00e+00 | pass |
| repetition_d5_r3 | 4 | 2 | 3 | 0.00e+00 | 0.00e+00 | pass |
| coherent_d3_r1 | 12 | 2 | 3 | 0.00e+00 | 0.00e+00 | pass |

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

Sensitivity: the tolerance column gives the range of that per-column bound, so
a smaller deviation would go unnoticed in that row. A row whose maximum
deviation is 0 has identical column means in both versions: the seeded error
draws coincide there, so the row shows that both versions compute the same
outcomes from the same draws rather than testing the statistics.

| Circuit | Columns | Shots (old / new) | Tolerance | Max abs deviation | Worst z | Result |
|---|---:|---:|---:|---:|---:|---|
| surface_d3_r2 | 17 | 200,000 / 200,000 | 1.4e-03 to 2.7e-03 | 0.00e+00 | 0.00 | pass |
| repetition_d5_r3 | 17 | 200,000 / 200,000 | 1.7e-03 to 2.8e-03 | 0.00e+00 | 0.00 | pass |
| surface_d7_r7 | 337 | 200,000 / 200,000 | 1.2e-03 to 5.0e-03 | 0.00e+00 | 0.00 | pass |
| coherent_d3_r1 | 9 | 200,000 / 200,000 | 1.1e-03 to 1.3e-03 | 0.00e+00 | 0.00 | pass |
| coherent_d5_r1 | 25 | 200,000 / 200,000 | 1.1e-03 to 1.7e-03 | 0.00e+00 | 0.00 | pass |
| distillation | 45 | 20,000 / 20,000 | 1.8e-02 to 2.5e-02 | 8.75e-03 | 2.40 | pass |
| msc_d3 | 21 | 20,000 / 20,000 | 4.2e-03 to 1.7e-02 | 4.35e-03 | 1.87 | pass |

## Benchmarks

Circuits: `clifft()` with noise, detector mode. Compile time is one
`compile_detector_sampler(strategy=..., seed=0)` call. Terms is the sum of
`num_graphs` over every compiled scalar graph of every component (0 when all
outputs take the direct path). After one warm-up `sample(4096)` call, one
`sample(65536, batch_size=8192)` call is timed (`sample(4096, batch_size=64)`
for msc_d3). Ratios are new/old: above 1 means the new compile is slower, and
below 1 means the new sampling is slower. Each run benchmarks one version and
then the other on a shared machine (the load column is the 1-minute load
average after each circuit, old / new). A throughput ratio marked `*` comes
from a timed call under 10 ms; it is noisy and not counted in
the verdict. Unlike the clifft-bench harness, no postselection mask is passed,
so absolute rates differ from that harness's. Benchmarks are reported, not
gated.

Not compared: the other three of clifft-bench's eight Tsim workloads,
`msc_d5_inject_cultivate_p1e-3` (cutting), `coherent_d3_r3` and `coherent_d5_r5`
(cat5). `clifft()` leaves them out because they exceed clifft-bench's 120 s
compile limit with today's Tsim, on clifft-bench's published H100 run and in
the local CPU calibration (`benchmark_instructions.md`).

| Circuit | Run | Strategy | Compile s (old / new) | New/old | Terms (old / new) | Shots/s (old / new) | New/old | Load (old / new) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| surface_d7_r7 | 1 | cat5 | 2.56 / 0.937 | 0.37 | 0 / 0 | 1,182,808 / 1,319,034 | 1.12 | 12.1 / 14.0 |
| surface_d7_r7 | 2 | cat5 | 2.53 / 0.937 | 0.37 | 0 / 0 | 1,169,978 / 1,406,500 | 1.20 | 14.5 / 15.8 |
| surface_d7_r7 | 3 | cat5 | 2.64 / 0.939 | 0.36 | 0 / 0 | 943,268 / 1,299,703 | 1.38 | 12.6 / 13.5 |
| coherent_d3_r1 | 1 | cat5 | 0.0244 / 0.0195 | 0.80 | 0 / 0 | 326,184,423 / 310,353,014 | 0.95\* | 12.1 / 14.0 |
| coherent_d3_r1 | 2 | cat5 | 0.0266 / 0.0183 | 0.69 | 0 / 0 | 289,769,483 / 321,188,854 | 1.11\* | 14.5 / 15.8 |
| coherent_d3_r1 | 3 | cat5 | 0.0245 / 0.0184 | 0.75 | 0 / 0 | 275,844,710 / 215,196,006 | 0.78\* | 12.6 / 13.5 |
| coherent_d5_r1 | 1 | cat5 | 0.0522 / 0.0325 | 0.62 | 0 / 0 | 83,841,434 / 94,517,395 | 1.13\* | 12.1 / 14.0 |
| coherent_d5_r1 | 2 | cat5 | 0.0543 / 0.0303 | 0.56 | 0 / 0 | 85,305,568 / 90,545,349 | 1.06\* | 14.5 / 15.8 |
| coherent_d5_r1 | 3 | cat5 | 0.0535 / 0.0308 | 0.57 | 0 / 0 | 86,373,639 / 75,702,109 | 0.88\* | 12.6 / 13.5 |
| distillation | 1 | cat5 | 1.19 / 0.728 | 0.61 | 520 / 446 | 86,716 / 160,729 | 1.85 | 12.9 / 13.5 |
| distillation | 2 | cat5 | 1.20 / 0.701 | 0.58 | 520 / 446 | 76,939 / 155,542 | 2.02 | 17.5 / 15.4 |
| distillation | 3 | cat5 | 1.22 / 0.743 | 0.61 | 520 / 446 | 83,001 / 158,505 | 1.91 | 11.8 / 12.8 |
| msc_d3 | 1 | cutting | 7.47 / 7.69 | 1.03 | 3,873 / 3,873 | 344 / 1,726 | 5.02 | 14.0 / 14.9 |
| msc_d3 | 2 | cutting | 7.60 / 7.65 | 1.01 | 3,873 / 3,873 | 361 / 1,718 | 4.76 | 16.9 / 13.7 |
| msc_d3 | 3 | cutting | 7.47 / 8.04 | 1.08 | 3,873 / 3,873 | 361 / 1,455 | 4.04 | 13.5 / 13.5 |

\* Timed call under 10 ms; noisy, not counted in the verdict.

## Tsim imported by each run

| File (old / new) | Old `tsim.__file__` | New `tsim.__file__` |
|---|---|---|
| probabilities.json / probabilities.json | `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/baseline-3dacc1a/src/tsim/__init__.py` | `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/zx-param-migration/src/tsim/__init__.py` |
| rates.json / rates.json | `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/baseline-3dacc1a/src/tsim/__init__.py` | `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/zx-param-migration/src/tsim/__init__.py` |
| benchmark-1.json / benchmark-1.json | `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/baseline-3dacc1a/src/tsim/__init__.py` | `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/zx-param-migration/src/tsim/__init__.py` |
| benchmark-2.json / benchmark-2.json | `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/baseline-3dacc1a/src/tsim/__init__.py` | `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/zx-param-migration/src/tsim/__init__.py` |
| benchmark-3.json / benchmark-3.json | `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/baseline-3dacc1a/src/tsim/__init__.py` | `/Users/rafaelhaenel/Documents/quera/tsim/.worktrees/zx-param-migration/src/tsim/__init__.py` |
