# Trajectory sharing for strategy="active_statevector": investigation

- **Date:** 2026-10-11
- **Machine:** Apple M4 Pro, 14 cores (10 performance, 4 efficiency), JAX 0.9.2 CPU
- **Branch:** `trajectory-sharing`, from `statevector-sampler` at `883b125`
- **Circuits:** the five clifft-bench workloads of `scripts/statevector/benchmark.py`,
  postselected on every detector as clifft-bench does
- **Code:** `src/tsim/statevector/shared.py` (the prototype, behind
  `trajectory_sharing=True`), `scripts/trajectory/` (measurements)

Shots whose quantum trajectories agree can share one state vector. The questions:

1. How much sharing do the benchmark circuits allow?
2. Does it fit an executor that needs static array shapes (JAX)?
3. What does it gain end to end?
4. Does Tsim's way of posing the problem (ZX reduction, an extracted circuit that
   measures the detectors directly) help?

## Summary

- **There is a lot of sharing.** In all five circuits the only per-shot input to
  the amplitudes is the sign of a rotation, and that sign depends on noise bits
  alone (one rotation in distillation also depends on an outcome). Once the
  classical detectors are postselected, the surviving shots share a handful of
  trajectories: one state vector per shot does 700× to 12,000× more array work
  than needed. coherent_d5_r5 is the exception (3×), because coherent noise lets
  many outcomes occasionally flip; dropping a branch as soon as it fires a
  postselected detector collapses it to 5 surviving histories (617× less work).
- **It works with static shapes.** The prototype keeps lanes in fixed-capacity
  tables with a traced count, splits them with prefix sums and scatters, exits
  the loop when a table is full and lets the host double it and resume, and runs
  wide components in segments so the host can shrink the table to the live
  lanes. The costs are compilations (one per table capacity) and host syncs.
- **End to end:** 78× on coherent_d5_r5 (890 → 69,019 shots/s, 2.7× Clifft 0.11
  on all threads), 3.2× on coherent_d3_r3, 2.2× on cultivation_d3, 2.0× on
  cultivation_d5, no change on distillation. On the small circuits the shared
  executor itself is 5× to 12× faster than one lane per shot, but the classical
  per-shot work (noise sampling, direct detectors, survivor buffering) now
  dominates the call.
- **Tsim's representation helps a lot, mostly through the classical/quantum
  split.** Postselecting on the classical (direct) detectors before any quantum
  simulation multiplies the sharing by 34× to 195× on the cultivation circuits
  and removes all noise-driven branching on coherent_d3_r3. The reduction also
  turns random measurement outcomes into classical bits, so noiseless
  cultivation has 2 trajectories in 16,384 shots. Sampling detectors rather than
  raw records matters mostly because it makes that postselection possible; by
  itself it changes little (2× on coherent_d3_r3). Nothing in the representation
  removes genuinely quantum branching (coherent_d5_r5); only pruning does.
- **Correctness:** an independent review confirmed the exactness argument and
  found four problems (reference-sample pruning, float32 resolution of rare
  outcomes, resume determinism, leftover values in pruned rows), all fixed and
  tested.

## 1. How much sharing the circuits allow

`sharing_stats.py` runs the existing executor action by action and counts, after
every array action, the distinct histories (rotation signs and active outcomes
so far: the lanes a splitting executor needs) and the distinct amplitude arrays.
A work ratio is the array work of one lane per shot divided by the work of one
lane per history (or state); "+ pruning" drops a lane once one of its
postselected records is 1.

Shots that survive postselection on every direct detector (the benchmark):

| circuit | lanes | final histories | final states | largest final lane | work ratio, histories | work ratio, states | work ratio, histories + pruning |
|---|---:|---:|---:|---:|---:|---:|---:|
| distillation | 65,536 | 32 | 7 | 13.1% | 11,703 | 16,949 | 11,703 |
| cultivation_d3 | 65,536 | 16 | 4 | 85.3% | 5,958 | 8,371 | 7,447 |
| coherent_d3_r3 | 65,536 | 664 | 664 | 39.8% | 724 | 724 | 4,446 |
| cultivation_d5 | 65,536 | 79 | 4 | 85.5% | 1,379 | 2,598 | 2,362 |
| coherent_d5_r5 | 16,384 | 14,652 | 14,635 | 1.0% | 3 | 3 | 617 |

The ratios grow with the number of shots (coherent_d3_r3: 103 at 4,096, 264 at
16,384, 724 at 65,536), since the number of distinct histories grows much slower
than the number of shots.

All shots, as a decoder-based workflow would sample them:

| circuit | lanes | final histories | final states | work ratio, histories | work ratio, states |
|---|---:|---:|---:|---:|---:|
| distillation | 65,536 | 32 | 7 | 11,703 | 16,949 |
| cultivation_d3 | 65,536 | 776 | 4 | 250 | 523 |
| coherent_d3_r3 | 65,536 | 1,084 | 1,084 | 319 | 319 |
| cultivation_d5 | 65,536 | 17,055 | 4 | 10 | 35 |
| coherent_d5_r5 | 16,384 | 14,837 | 14,823 | 3 | 3 |

Histories now branch on noise, but cultivation_d5's 17,055 histories end in 4
distinct states: there, merging lanes that reach the same state would recover
most of the sharing.

## 2. Does Tsim's representation help?

Tsim does not simulate the circuit as written. It reduces the noisy circuit's ZX
diagram, moves every output that equals a single error-basis bit to a classical
("direct") path, rewrites the noise in a basis found by Gaussian elimination,
and extracts a circuit for each remaining connected component whose outputs are
the detectors and observables themselves. A frame simulator such as Clifft
keeps the whole circuit and computes detectors from measurement records.

### What the representation hands the executor

`structure.py`, for detector sampling (the benchmark) and for raw measurement
records:

| circuit | sampling | raw error bits | basis bits | component f-bits | f-bits in rotation signs | key rank | direct / all outputs | quantum width | active / dormant measurements |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| distillation | detectors | 792 | 46 | 6 | 0 | 0 | 40 / 45 | 3 | 5 / 0 |
| distillation | records | 792 | 46 | 46 | 0 | 0 | 0 / 85 | 3 | 5 / 40 |
| cultivation_d3 | detectors | 1,157 | 45 | 38 | 17 | 17 | 16 / 21 | 5 | 6 / 1 |
| cultivation_d3 | records | 1,157 | 45 | 33 | 17 | 17 | 15 / 21 | 5 | 6 / 1 |
| coherent_d3_r3 | detectors | 74 | 48 | 31 | 19 | 19 | 17 / 25 | 5 | 12 / 4 |
| coherent_d3_r3 | records | 74 | 52 | 40 | 23 | 19 | 12 / 33 | 4 | 8 / 8 |
| cultivation_d5 | detectors | 7,871 | 194 | 173 | 75 | 57 | 93 / 108 | 6 | 16 / 2 |
| cultivation_d5 | records | 7,871 | 194 | 118 | 57 | 57 | 85 / 112 | 6 | 16 / 2 |
| coherent_d5_r5 | detectors | 314 | 214 | 141 | 81 | 81 | 73 / 121 | 12 | 60 / 12 |
| coherent_d5_r5 | records | 314 | 226 | 166 | 92 | 81 | 60 / 145 | 15 | 48 / 24 |

"key rank" is the rank of the rotation signs' f-parts: the grouping key without
pruning. In detector mode, the postselected detectors sit on average at 57%
(cultivation_d5) to 81% (cultivation_d3) of the way through the tape, and 12% to
43% of them in its first half.

### What each piece is worth

Work ratio at 16,384 shots (coherent_d5_r5 records: 2,048 shots at width 15):

| circuit | records, all shots | detectors, all shots | detectors, postselected | + pruning | detectors, noiseless |
|---|---:|---:|---:|---:|---:|
| distillation | 2,926 | 2,926 | 2,926 | 2,926 | 2,926 |
| cultivation_d3 | 110 | 112 | 3,773 | 7,447 | 13,336 |
| coherent_d3_r3 | 75 | 149 | 264 | 1,718 | 264 |
| cultivation_d5 | 7.3 | 7.4 | 1,442 | 3,059 | 10,889 |
| coherent_d5_r5 | 1.5 | 3.2 | 3.4 | 617 | 3.4 |

Without noise, records branch as little as detectors on the cultivation circuits
(2 histories in 16,384 shots in both modes) and more on the coherent ones
(coherent_d3_r3: 1,209 against 399).

1. **Classical detectors decided before any quantum simulation: the largest
   effect.** The reduction separates every detector that depends on noise alone,
   wherever it sits in the circuit, and the sampler evaluates it from the noise
   bits first. Most noise that would flip a rotation sign also fires such a
   detector, so postselection removes the shots that would split trajectories
   before they start: 34× more sharing on cultivation_d3, 195× on cultivation_d5,
   and on coherent_d3_r3 the postselected ratio equals the noiseless one (all
   noise-driven branching is gone). It also removes work outright: only 8%
   (distillation) to 96% (coherent_d3_r3) of the shots reach the quantum stage.
2. **Random outcomes become classical bits.** In Tsim a uniformly random
   measurement outcome is an error-basis bit with probability ½ on the direct
   path, even when sampling raw records (cultivation_d5: 85 of 112 records are
   direct). So gauge randomness never becomes a branch point: noiseless
   cultivation has 2 trajectories whichever outputs are sampled.
3. **Measuring detectors directly matters mainly through point 1.** Records
   cannot be postselected before simulating, so that factor is lost entirely.
   Without postselection, detectors share as well as records on the cultivation
   circuits and 2× better on coherent_d3_r3 (3× fewer final histories without
   noise), likely because a detector is one parity of records whose individual
   outcomes branch more. The extracted
   tape writes each detector at its own measurement, which is what makes pruning
   possible, but the detectors come late in the tapes, so a scheduler that pulls
   postselected detectors earlier would prune earlier.
4. **A small linear key.** Noise reaches the amplitudes only as parities of a
   few error-basis bits: 7,871 raw error bits become a 57-bit key on
   cultivation_d5, 314 become 81 on coherent_d5_r5 (71 and 129 bits once the
   postselected signs join for pruning). Shots group by those parities, so many
   raw error patterns share a lane.
5. **Narrow lanes.** Reduction and scheduling give width 6 against Clifft's 10
   on cultivation_d5 (16× fewer amplitudes per lane) and 12 against 13 on
   coherent_d5_r5, where hundreds of lanes remain alive.
6. **What it cannot remove.** coherent_d5_r5 branches as much without noise as
   with it (14,652 histories in 16,384 shots): coherent rotations make the
   detector outcomes themselves random. Only pruning helps there.

### Compared with a frame simulator

This part is structural; I did not implement sharing in Clifft. Clifft's
frame-factored state has the same core property: noise acts through the Pauli
frame and reaches the active state only by flipping a later rotation's sign, so
grouping shots by the frame's effect on rotation signs would work there too.
Three things are harder without Tsim's representation:
- **Postselection happens in circuit order.** Clifft aborts a shot at its first
  failing postselected detector while simulating, so a noisy shot splits a
  trajectory until that detector comes; Tsim filters it out before the
  trajectory starts.
- **Gauge randomness enters the frame per shot.** A random dormant outcome
  leaves the active state alone but enters the frame, so it splits trajectories
  wherever it reaches a later rotation sign, unless the compiler proves it does
  not; the reduction has removed it.
- **Lanes are wider.** Each lane would carry 2^10 rather than 2^6 amplitudes on
  cultivation_d5.

### Verdict

Yes, significantly: the classical/quantum split of the ZX reduction is what
turns "noisy shots" into "a handful of trajectories" (1,442× instead of 7.4× on
cultivation_d5 at 16,384 shots), and the reduction keeps quantum branching at its
intrinsic minimum. Sampling detectors rather than records is necessary for that
postselection but adds little by itself.

## 3. Where the time goes with one lane per shot

`breakdown.py`, per benchmark call (shots per call and batch size of the
statevector benchmark):

| circuit | survival | noise s | direct s | statevector s | call s | statevector share of the call |
|---|---:|---:|---:|---:|---:|---:|
| distillation | 0.079 | 0.079 | 0.017 | 0.023 | 0.249 | 9% |
| cultivation_d3 | 0.704 | 0.008 | 0.008 | 0.424 | 0.759 | 56% |
| coherent_d3_r3 | 0.959 | 0.002 | 0.016 | 0.388 | 0.510 | 76% |
| cultivation_d5 | 0.158 | 0.023 | 0.014 | 0.058 | 0.109 | 53% |
| coherent_d5_r5 | 0.844 | 0.002 | 0.002 | 18.530 | 18.559 | 100% |

So sharing alone can at most give 1.1× on distillation, about 2× on the
cultivation circuits and 4× on coherent_d3_r3, unless the rest of the call gets
faster too. "call" exceeds the sum of the stages by up to 0.32 s (cultivation_d3);
that remainder is the postselection path itself, which buffers the surviving
shots as a Python list of rows before stacking them into batches.

## 4. The prototype

`src/tsim/statevector/shared.py`, enabled with
`compile_detector_sampler(strategy="active_statevector", trajectory_sharing=True)` (or
`compile_sampler`).

- **Grouping key.** A shot's key is a GF(2) basis of the f-bit parts of every
  rotation sign, and with postselection of every postselected output's sign (so a
  lane can be pruned as a whole). A measurement's sign never reaches the
  amplitudes: the kernels project on the unsigned outcome and only the record is
  flipped. Shots are sorted by key with random tie-breaking; each group is a
  contiguous range in random order.
- **Lanes.** One lane per group: amplitudes, the symbols of one member, a shot
  count and a range. At a measurement the lane draws how many of its shots see
  the rarer outcome and splits; the child takes the last n shots of the range.
  Whether any shot sees the rarer outcome is decided like the per-shot
  executor's `choose_outcome`, at 2^-64 resolution; given one, the first such
  shot and the rest are drawn exactly. A dormant measurement whose outcome
  reaches a key sign splits the same way with p = ½; the others are coin flips
  drawn per shot at the end. A lane whose postselected record (XOR the detector
  reference bit, with `use_detector_reference_sample`) is 1 moves to a dead table
  that keeps its range, outcomes and pruning action but no amplitudes.
- **Records.** After the tape each shot evaluates every output from its own
  f-bits, its lane's outcomes and its own dormant coins, and the results are put
  back in the original shot order. A pruned shot's records after its pruning
  record are False.
- **Exactness.** Shots of a lane agree on every key bit, so their states agree
  given the outcomes, and outcomes are independent given the state. Sequential
  binomial splits give the multiset of outcome histories of independent draws,
  and because each range is in uniformly random order, handing the last n shots
  to the child is a uniformly random assignment. (Without the random tie-break,
  shots that are neighbours in the original order get correlated outputs; the
  per-shot distributions stay right, but a test that compared blocks of shots
  caught it.)

### Static shapes

Every array has a static shape; the number of live lanes is a traced scalar.

1. **Fixed-capacity tables.** Splits allocate child slots with a prefix sum and a
   scatter (`mode="drop"`), pruning compacts the live lanes with another, and every
   kernel runs over the whole table. The dead table has one slot per shot and
   never fills (it holds no amplitudes: about 1.5 MB at 65,536 shots).
2. **Overflow.** When a split would overflow the lane table, the action returns
   its input unchanged with a flag, the `while_loop` stops, and the host doubles
   the table and resumes at the same action. This is the `run(start, stop)` hook
   of the v1 design. Each lane slot draws from `fold_in(fold_in(key, a), slot)`,
   so the redone action draws the same numbers, and samples are bitwise equal
   whatever the capacities and segments (tested). One key for the whole table is
   not enough: JAX's binomial sampler loops over all slots until each accepts, so
   its draws depend on the table size.
3. **Segments.** Wide components (W ≥ 8) run 32 actions at a time; between
   segments the host shrinks the lane table to a power of two above the live
   lanes. Without it, coherent_d5_r5 computed 256 lanes when 2 were alive.
4. **Few capacities.** Every new capacity compiles the loop again (about a second
   at W = 6). Before the dead table was sized to the batch and narrow components
   kept their largest capacity, each benchmark call met new capacities and
   stalled for 1 to 2.5 s; cultivation_d5 measured 602k shots/s in place of 954k.
5. **Sorting on the host.** XLA's CPU sort takes 9.3 ms for 65,536 shots (two
   keys); a numpy permutation plus stable radix sorts take 0.6 ms. The key bits
   are computed on the device and sorted on the host. A GPU build would sort on
   the device. Reconstruction finds each shot's lane with a scatter and a running
   maximum rather than a sort.

### Review and tests

An independent review confirmed the exactness argument: the key covers every
input that reaches the amplitudes, a measurement's sign never reaches the state,
the positional assignment after a random tie-break is exact, and pruning,
resizing and the segmented path are correct. It found four problems, now fixed:

- pruning ignored `use_detector_reference_sample`, so a postselected detector
  with reference bit 1 pruned the shots the user keeps;
- `jax.random.binomial` draws float32 uniforms, so a lane cut the chance of a
  rare outcome down to a 2^-23 grid (a lone shot never saw an outcome of
  probability below about 1e-7, the range small coherent rotations produce);
- a resumed action could draw different numbers (see item 2 above);
- rows discarded by a component detector carried leftover values in later
  columns (now False).

Tests:

- `test/unit/statevector/test_shared.py` (41 tests): samples match the exact plan
  distributions for one noise pattern per batch, for mixed patterns in one batch,
  and with pruning (with and without reference bits); pruned rows are False after
  their pruning record; capacity-1 tables grow and resume; capacities and segments
  leave samples bitwise unchanged; rare-outcome counts keep their probability,
  including an outcome of probability 1e-7 for single shots.
- `test/integration/test_trajectory_sharing.py` (27 tests): detector, measurement
  and postselected sampling with and without sharing agree on the seven helper
  circuits, on random noisy circuits, and on a circuit with a direct detector that
  exercises pruning through the sampler, with and without the reference sample
  (the latter fails if reference bits are ignored).
- Every acceptance in the benchmark below agrees with Clifft 0.10.0 within 5
  standard errors.

## 5. Results

### The shared executor alone

`shared_breakdown.py`: one batch of shots that survived the direct detectors,
fastest of five runs. "lanes and checks" is the rest of the shared time: the lane
loop, its host checks and resizes.

| circuit | shots | key bits | groups | peak live / dead lanes | one lane per shot ms | group ms | lanes and checks ms | reconstruct ms | shared ms | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distillation | 65,536 | 0 | 1 | 32 / 0 | 6.6 | 0.6 | 1.2 | 0.7 | 2.5 | 2.6× |
| cultivation_d3 | 65,536 | 21 | 29 | 5 / 33 | 48.1 | 1.7 | 2.6 | 0.9 | 5.1 | 9.4× |
| coherent_d3_r3 | 65,536 | 27 | 29 | 19 / 227 | 92.1 | 1.8 | 4.9 | 1.0 | 7.6 | 12.1× |
| cultivation_d5 | 16,384 | 71 | 139 | 9 / 157 | 57.0 | 1.9 | 8.6 | 0.8 | 11.3 | 5.1× |
| coherent_d5_r5 | 4,096 | 129 | 90 | 174 / 589 | 4,552.1 | 0.9 | 152.8 | 0.4 | 154.1 | 29.5× |

The array work has nearly gone on the small circuits: what remains is fixed
per-shot cost (key bits, sort, reconstruction) and the lane loop over a table
sized for the initial groups. On coherent_d5_r5 the lane loop still dominates
(section 6).

### End to end

`benchmark_sharing.py`, 10 s per row, postselection on every detector, same seed
for every sampler. "shared" rows use the benchmark's batch size and a batch of
65,536; a shared batch needs no state vector per shot, so it can be large.

| circuit | one lane per shot | shared | shared, batch 65,536 | best speedup | Clifft 0.11, 1 thread | Clifft 0.11, all threads |
|---|---:|---:|---:|---:|---:|---:|
| distillation | 2,891,488 | 2,977,107 | 2,978,761 | 1.03× | 2,755,248 | 20,196,109 |
| cultivation_d3 | 1,084,971 | 2,341,123 | 2,271,734 | 2.2× | 4,742,046 | 30,405,370 |
| coherent_d3_r3 | 540,825 | 1,728,802 | 1,702,539 | 3.2× | 1,920,603 | 13,664,724 |
| cultivation_d5 | 555,858 | 976,865 | 1,095,363 | 2.0× | 136,869 | 1,207,485 |
| coherent_d5_r5 | 890 | 27,280 | 69,019 | 78× | 3,692 | 25,476 |

Measured at commit `88c99cd`. Clifft's figures are the same-machine measurements
in `scripts/statevector/results.md` of the main checkout (Clifft 0.11.0 from
PyPI). Acceptances (shared): 0.0799, 0.6866, 0.3732, 0.1437 to 0.1441 and 0.0076
to 0.0078, all "ok" against Clifft 0.10.0. Logical errors among accepted shots
are 0 except on distillation (rate 0.4220, Clifft 0.422) and cultivation_d3 (15
in 1.62e7, 9.3e-7; Clifft 0.11.0 measured 222 in 2.09e8, 1.1e-6). Warmup (first
call, JIT included) is 1.4 to 2.4 s with sharing on the small circuits and 5.6
to 12.6 s on coherent_d5_r5, against 0.6 to 1.0 s and 19 s with one lane per
shot.

## 6. What limits the gains, and what would come next

- **Small circuits are now limited by classical per-shot work.** A cultivation_d5
  call with sharing takes about 60 ms: noise sampling 37 ms
  (`ChannelSampler.sample` loops over the channels in Python, and the
  postselection path calls it once per batch, so small batches pay it several
  times), direct detectors 11 ms, survivor buffering a few ms, the shared
  statevector part about 10 ms. Further gains need those stages vectorized.
- **Wide circuits are limited by the grouping key.** The key holds every key bit
  from the start, so noise that only flips a late postselected detector gets its
  own lane from action 0: coherent_d5_r5 starts with 83 (4,096 shots) to 398
  (65,536 shots) groups and peaks at about 160 to 820 live lanes, while one lane
  per surviving history would average about 14 to 27. Splitting lazily, when a
  key bit is first needed, would need per-shot outcome draws: handing the last n
  shots of a range to one outcome is biased once a lane's shots differ in later
  key bits. That per-shot work is cheap next to 4,096 amplitudes per lane, and
  the statistics above suggest up to about 10× more on coherent_d5_r5; at W ≤ 6
  it would cost about as much as it saves.
- **Earlier detectors.** Postselected detectors sit late in the tapes (section
  2); scheduling them earlier would prune branches sooner.
- **Without postselection, merge states.** Histories branch on noise, but few
  distinct states remain (cultivation_d5: 4). Hashing amplitudes to merge equal
  lanes would recover most of the sharing.
- **Static shapes cost compilations and syncs.** Warmup grows with the number of
  capacities met; the host checks once per segment. On a GPU each sync and launch
  costs more, so fewer segments, device-side sorting and precompiled capacities
  matter more there.
- **Contract change.** With pruning, a row discarded by a component detector
  reports False after its pruning record instead of the full sample.
- **Prototype shortcuts.** The automatic batch size and the memory check still
  assume one state vector per shot, and `sharing_stats` is instrumentation.

## Reproduce

From the worktree root (with the main checkout's environment):

    uv run python scripts/trajectory/sharing_stats.py --lanes 4096 16384 65536 --max-amplitudes 67108864
    uv run python scripts/trajectory/sharing_stats.py --all-shots --lanes 4096 16384 65536 --max-amplitudes 67108864
    uv run python scripts/trajectory/sharing_stats.py --mode measurements --lanes 2048 16384 --max-amplitudes 67108864
    uv run python scripts/trajectory/sharing_stats.py --all-shots --noiseless --lanes 16384 --max-amplitudes 67108864
    uv run python scripts/trajectory/sharing_stats.py --mode measurements --noiseless --lanes 2048 16384 --max-amplitudes 67108864
    uv run python scripts/trajectory/structure.py
    uv run python scripts/trajectory/breakdown.py
    uv run python scripts/trajectory/shared_breakdown.py
    uv run python scripts/trajectory/benchmark_sharing.py --seconds 10
