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
- Tests: `test/unit/statevector/test_conditioned.py` (44); existing statevector
  suites pass.

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
| at most one piece per thread per key group | **0.355 us** | 0.074 us (noise) |

Noise sampling is ~0.02 us/shot. Speedup over `active_statevector`: ~1.6e8 (24 threads each).
Fault-free lanes have width 9 (= logical qubits); with the general Pauli compression
faulty lanes stay at <= 9 too (Z-type compression alone let them grow to 21).

Other runs (24 threads; 1 thread in brackets):

| circuit | postselected | all shots |
|---|---|---|
| BLP, 60 detectors | 0.068 us (0.355 us) | 9.6 us (158 us) |
| BLP, paper's 27 detectors | 0.26 us (2.6 us) | |
| toric [[81,3,3]], 168 detectors (`active_statevector`: k = 63, cannot run) | 0.34 us* | |
| toric, paper's 81 detectors | 2.85 us (40.7 us) | |

(* before the general Pauli compression.) Without postselection the shots whose
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
