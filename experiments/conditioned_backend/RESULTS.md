# `strategy="conditioned_statevector"` — results (Nov 6 2026)

Machine: Ryzen 9 9900X (12 cores / 24 threads), 60 GB. Branch
`experiment/conditioned-backend` off `main` (c454fa23).

## The backend

`src/tsim/statevector/conditioned/`: a lane (noise key, outcome branch) holds a
Clifford frame, a dense register of active qubits and pending T-like rotations.
A static step plan (per component) fixes which rotations each measurement applies
and which it *measures through*: the measurement of `M` behind a layer `Q` is the
measurement of the Clifford involution `O = Q^dag M Q`, whose projection is built
directly (dormant Z-strings that `O` preserves stay frozen; `O|0>` from its
stabilizers plus one Gauss-sum phase; `O|a,0>` from the support of `O|0>`).
Z-type stabilizers go back into the frame after each step. Lanes share shots by
key over a whole sampling call; outcomes come from per-shot counter-based
uniforms, so samples do not depend on threads, chunks or grouping.

Shortcut at the root: a faulty key's measure-through from `|0...0>` follows from
the reference key's outcome by Pauli algebra (`K = K0 (-1)^|F| P_F`).

## Correctness

- Gauss sums vs brute force (3000 random forms): exact.
- Engine vs dense statevector (outcome probabilities along random paths): <= 5e-14.
- `exact.key_leaves` (all branches of a key) vs dense enumeration (tricycle 6,
  fault-free and faulty keys): TV <= 1.5e-14.
- Sampler vs `active_statevector`: acceptance, detector marginals and 512-bin
  observable histograms agree (cultivation d3/d5, end2end d5, distillation,
  coherent d3r3, tricycle 6/9; helper circuits with and without postselection).
- Every optimization kept postselected samples bit-identical (hash check).
- Tests: `test/unit/statevector/test_conditioned.py` (46); the full suite passes
  (2538; `test_the_calling_thread_keeps_parallel_kernels_after_run_tasks` fails
  on origin/main too in this environment).
- Plan choice: each component tries the measure-through plan (op order) and a
  plan without measure-through in Tsim's scheduler order on the reference key,
  and keeps the narrower: distillation width 2 (Tsim 3; measure-through 5),
  coherent d3r3 3 (Tsim 4); factories and cultivation keep measure-through.
  Distillation vs `active_statevector` at 1e7 shots each: p = 0.90.

## Sampling time per shot (BLP [[27,3,3]] factory, postselected on all detectors)

`blp_27_3_3.stim` (60 detectors), Tsim's `active_statevector`: k = 26, **11.2 s per
shot** (64 shots, 24 threads, no postselection).

| version | 1 thread | 24 threads |
|---|---|---|
| first working backend | 75 us | 9.3 us |
| O\|a,0> from the support of O\|0> (2^(w+r), not 2^(w+w')) | 4.1 us | 0.46 us |
| per-lane cache of op images (frame version row) | 3.6 us | |
| O's images by transposed masks, updated through the CX network | 2.8 us | 0.51 us |
| root shortcut for faulty keys; numba key grouping | 1.9 us | 0.15 us |
| group by key over the whole call (one tree per key) | 0.75 us | 0.18 us |
| fault-free shots unsorted; big groups cut into pieces | 0.77 us | 0.099 us |
| general Pauli-stabilizer compression (Walsh-transform candidates) | 0.425 us | **0.068 us** |
| at most one piece per thread per key group | 0.355 us | 0.070 us |
| reference key's outcome tree cached on the component | **0.345 us** | **0.061 us** |

Noise sampling is ~0.02 us/shot. Speedup over `active_statevector`: ~1.6e8 (24 threads each).
Fault-free lanes have width 9 (= logical qubits); with the general Pauli compression
faulty lanes stay at <= 9 too (Z-type compression alone let them grow to 21).

Other runs (24 threads; 1 thread in brackets):

| circuit | postselected | all shots |
|---|---|---|
| BLP, 60 detectors | 0.061 us (0.345 us) | 9.6 us (158 us) |
| BLP, paper's 27 detectors | 0.25 us (2.5 us) | |
| toric [[81,3,3]], 168 detectors (`active_statevector`: k = 63, cannot run) | 0.20 us (2.0 us) | |
| toric, paper's 81 detectors | 2.85 us (40.7 us) | |

Per call (BLP, 24 threads): calls of 1e3 / 1e4 / 1e5 / 1e6 shots take
5.3 / 8.9 / 14 / 62 ms (5.3 / 0.89 / 0.14 / 0.062 us per shot); before the tree
cache every call rebuilt the fault-free lanes (22 ms).

Against `active_statevector` on the benchmark circuits (postselected, 24 threads, 2e6 shots):

| circuit (Tsim k) | Tsim | conditioned |
|---|---|---|
| cultivation d3 (4) | 0.008 us | 0.012 us |
| cultivation d5 (6) | 0.007 us | 0.015 us |
| end2end d5 (6) | 0.083 us | 0.100 us |
| distillation (3) | 0.006 us | 0.010 us |
| coherent d3r3 (4) | 0.007 us | 0.011 us |
| tricycle 6 (9) | 0.028 us | 0.025 us |
| tricycle 9 (11) | 1.71 us | 0.030 us (57x) |
| tricycle 12 (16) | 388 us | 0.033 us (11,700x) |
| BLP 27 (26) | 11.2 s (no postselection) | 0.061 us |

At small k both are bound by the noise sampling and Tsim is up to 2x faster.

Without postselection the shots whose
detectors fire run to the end, and every outcome of every shot is a new lane;
Tsim's `active_statevector` needs 11.2 s per BLP shot either way.

A lane wider than 26 qubits raises a ValueError suggesting a postselection mask
or `active_statevector`; with Z-type compression only, unpostselected BLP hit it
(lanes grew ~1 qubit per random detector); with the general compression none
were seen in 2e5 unpostselected shots.

## Literature: Li, Preskill, Xu, arXiv:2510.07269 (CCZ error detection)

The paper simulates the factory in Stim with the CCZ omitted and its noise as
weight-5 Z errors; detectors only on the final X checks (27 for BLP).

| circuit | acceptance | postselected logical error |
|---|---|---|
| paper, BLP d3 (1e9 shots) | 0.7041037 | 1.944e-6 +- 0.053e-6 |
| paper circuit re-sampled (Stim, 2e8) | 0.70406 +- 0.00003 | 1.939e-6 +- 0.12e-6 |
| ours, paper detectors, CCZ removed (Stim, 2e8) | 0.70409 +- 0.00003 | 2.04e-6 +- 0.12e-6 |
| **ours, paper detectors, real CCZ (2.1e8)** | **0.62131 +- 0.00003** | **2.307e-4 +- 0.010e-4** |
| ours, 60 detectors, CCZ removed (Stim, 2e8) | 0.58590 +- 0.00004 | 3.0e-7 +- 0.5e-7 |
| **ours, 60 detectors, real CCZ (1e9)** | **0.57997 +- 0.00002** | **6.39e-7 +- 0.31e-7** |
| paper, toric d3 [[81,3,3]] (5e8) | 0.34906 | 1.93e-6 +- 0.11e-6 |
| ours, toric, paper detectors, CCZ removed (Stim, 8.8e7) | 0.34903 +- 0.00005 | 2.12e-6 +- 0.26e-6 |
| **ours, toric, paper detectors, real CCZ (2e8)** | **0.23395 +- 0.00003** | **4.74e-5 +- 0.09e-5** |
| ours, toric, 168 detectors, CCZ removed (Stim, 9.6e7) | 0.20110 +- 0.00004 | 2.6e-7 +- 1.2e-7 |
| **ours, toric, 168 detectors, real CCZ (1e9)** | **0.19510 +- 0.00001** | **8.8e-8 +- 2.0e-8** |

Our circuits are the authors' (same Z-syndrome round, CCZ triples and noise) with a
feed-forward X correction after the random Z-syndrome round, real CCZ gates
(Tsim syntax), and optionally 33 (BLP) / 87 (toric) detectors on the
Z-syndrome outcomes.

Logical error with a real CCZ: the observables are random even ideally, so a
shot's error is the total variation distance between its configuration's
observable distribution and the ideal one; the estimator (`mc_lfr.py`) weights
each accepted shot by `max(0, 1 - D_0(o)/D_c(o))`, where the backend knows `D_c(o)`
exactly (product of the observable-step probabilities). For a deterministic
ideal this is exactly the usual flip rate. Checked against exact per-
configuration enumeration (`exact_lfr.py`) on tricycle 6: 1.471e-3 +- 0.009e-3
vs 1.476e-3.

Ablation (paper detectors, real CCZ, 5e7 shots each): without the Z-ancilla
depolarizing noise 0.6613 / 9.2e-5; without any noise before the CCZ
0.8737 / 0 events. The ~100x higher logical error of the paper's detector set is
first order in p: a syndrome-round fault gives a wrong feed-forward X correction
(or an X error) on data, which the real CCZ turns into CZ-type errors that the
final X checks miss. The Clifford proxy cannot see this; detectors on the
Z-syndrome outcomes (our 60-detector variant) bring it back to ~2x the proxy.

Reproduce: `mc_lfr.py CIRCUIT SHOTS`, `proxy_stim.py CIRCUIT SHOTS PROCS`,
`scaling.py N`.

## Logical error rates vs `active_statevector` (Nov 7)

`ler_all.sh` (`ler.py`), independent seeds per backend. Deterministic observables:
postselect every detector (cultivation) or decode with PyMatching on the DEM of
the circuit with T -> S and coherent rotations dropped (same decoder for both).

| circuit | shots | acceptance (active / conditioned) | logical error (active / conditioned) | z |
|---|---|---|---|---|
| cultivation d3 | 2e9 | 0.68678 / 0.68679 | 9.06e-7 / 8.77e-7 (1244 / 1204) | +0.8 |
| cultivation d5 | 4e9 | 0.14395 / 0.14396 | 6.9e-9 / 6.9e-9 (4 / 4) | 0.0 |
| coherent d3 r1 (decoded) | 5e7 | 1 / 1 | 7.54e-5 / 7.48e-5 (3770 / 3740) | +0.4 |
| coherent d3 r3 (decoded) | 5e7 | 1 / 1 | 3.72e-5 / 3.56e-5 (1862 / 1781) | +1.3 |
| coherent d5 r1 (decoded) | 5e7 | 1 / 1 | 1.72e-6 / 1.92e-6 (86 / 96) | -0.7 |

Random ideal observables (end2end d5: T state read out, P(1) = 0.853 noiseless;
SymFT d7 fires detectors noiselessly in Tsim; distillation, tricycle factories):
acceptance and observable histogram among accepted shots.

| circuit | shots | acceptance (active / conditioned) | z | histogram chi2 p |
|---|---|---|---|---|
| end2end d5 (paper mask) | 2e8 | 0.11697 / 0.11702 | -1.5 | 0.22 |
| SymFT d7 | 2e8 | 0.00554 / 0.00554 | -0.7 | 0.61 |
| distillation | 2e8 | 0.07979 / 0.07975 | +1.3 | 0.58 (32 patterns) |
| tricycle 6 | 1e8 | 0.88882 / 0.88882 | -0.1 | 0.27 (512) |
| tricycle 9 | 1e8 | 0.83590 / 0.83595 | -0.9 | 0.20 (512) |
