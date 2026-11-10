---
name: tsim
description: Simulate noisy quantum circuits that contain non-Clifford gates (T, T_DAG, arbitrary rotations, U3, CCZ, CS) with Tsim (the bloqade-tsim package), a Stim-compatible sampler. Use this skill whenever a task mentions Tsim or bloqade-tsim, asks to sample measurements, detectors or observables from a Clifford+T or Clifford+rotation circuit, asks whether such a circuit can be simulated or how long it will take, or involves magic state distillation, cultivation or coherent errors in a Stim-style circuit, even if Tsim is not named.
license: Apache-2.0
compatibility: Python 3.11+. Written for Tsim main; `sampler.active_width` and `sampler.stats` need a release newer than 1.0.0.
metadata:
  author: QuEra Computing
  version: "1.1"
---

# Tsim

Tsim samples noisy quantum circuits with non-Clifford gates. It reads Stim's circuit format plus
a few non-Clifford instructions and mirrors Stim's sampler API, so most Stim code ports by
changing `stim` to `tsim`. Install with `uv add bloqade-tsim` or `pip install bloqade-tsim`;
import as `tsim`. Docs: https://bloqade.quera.com/tsim (append `.md` to any page for Markdown).

For logical error rates, decoders, postselection and fidelity, also read the `tsim-qec-study`
skill. For running circuits published with a paper, read `tsim-port-circuits`.

## Which circuits Tsim can simulate

Tsim's cost is set by one number, the **active width** w: the largest number of qubits that
must be held in a state vector at once, after Tsim has removed everything Clifford it can.
Each shot costs roughly 2^w operations. The number of T gates and the number of qubits do
not set the cost. End-to-end magic state cultivation at distance 5 (463 qubits, 53 T gates)
has width 6, a distance-5 surface code with coherent rotations on every qubit (64 qubits,
over 1,000 rotations) has width 11, and a 20-qubit quantum volume circuit has width 20.

Compile first, then read the width. Compilation is usually fast; sampling is where time goes.

```python
import tsim

c = tsim.Circuit(
    """
    R 0 1
    H 0
    T 0
    CNOT 0 1
    DEPOLARIZE2(0.01) 0 1
    H 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
    OBSERVABLE_INCLUDE(0) rec[-1]
    """
)
sampler = c.compile_detector_sampler(seed=1)
print(sampler.active_width)  # int; 0 means no state vector is needed
print(sampler.stats)  # component widths, peak bytes per shot, ...
dets, obs = sampler.sample(shots=100_000, separate_observables=True)
```

| Active width | What to expect on a typical multi-core CPU |
|---|---|
| 0 | No state vector; as fast as sampling the noise |
| up to about 16 | Comfortable; millions of shots are practical |
| about 20 to 26 | Slow; near 26 a shot takes on the order of seconds |
| above 30 | Rejected with a `ValueError` (or earlier, if memory runs out) |

Before a long run, time it: one small warm-up call (the first call compiles kernels), then
one call of about 10^6 shots, and extrapolate. Small calls understate the rate. Report the
width and the measured shots per second, with the thread count, along with any plan.

Why the width is often small: ZX simplification absorbs the Clifford gates, so they do not
appear in the program that runs on the state vector; only the non-Clifford rotations and the
measurements that interact with them do. Adding Clifford structure around a non-Clifford core
therefore leaves the width unchanged. Distance-5 cultivation has width 6 both as a 42-qubit
inject-and-cultivate circuit and as a 463-qubit end-to-end circuit with a surface-code escape
stage. The extra cost then moves elsewhere: the end-to-end circuit has 2,744 detectors, 2,730
of them sampled directly from the noise, and once postselection has removed most of the
state-vector work, that direct sampling is the bottleneck (3.3 vs 53.6 million shots/s for
the smaller circuit on 4 threads). For such circuits `bit_packed=True` helps, and the cost
grows with the number of detectors, not with the width.

If the width is too large: postselect inside Tsim (discarded
shots stop early, see below), simulate a smaller instance (lower distance or fewer rounds) and
report how the width grows, or check whether part of the circuit can be a Clifford stand-in.

## Circuit format

Every Stim instruction works, plus these. Angles are in units of π, not radians:
`R_Z(0.25)` is a rotation by π/4.

| Instruction | Meaning |
|---|---|
| `T`, `T_DAG` | π/4 phase gate and its inverse |
| `R_X(a)`, `R_Y(a)`, `R_Z(a)` | rotation by a·π about X, Y, Z |
| `U3(theta, phi, lambda)` | general single-qubit unitary |
| `R_XX(a)`, `R_YY(a)`, `R_ZZ(a)` | exp(−i a·π/2 · P) on two qubits |
| `R_PAULI(a) X0*Y1*Z2` | exp(−i a·π/2 · P) on a Pauli product |
| `TPP`, `TPP_DAG` | exp(∓iπ/8 · P) on a Pauli product |
| `CS`, `CS_DAG`, `CCZ`, `CCX` | controlled-S and Toffoli family |

Ways to build a circuit: `tsim.Circuit(text)`, `tsim.Circuit.from_file(path)`,
`tsim.Circuit(stim_circuit)`, or `c.append("R_Z", [0], 0.25)`.

Inside Stim text, Tsim writes non-Clifford gates as tagged Clifford instructions: `T` is
`S[T]`, `T_DAG` is `S_DAG[T]`, and `R_Z(0.25)` is `I[R_Z(theta=0.25*pi)]`. `c.stim_circuit`
returns that `stim.Circuit`. Stim reads it as a Clifford stand-in (T as S, rotations as
identity), which is useful for cross-checks. Many `stim.Circuit` methods are not on
`tsim.Circuit`; call them on `c.stim_circuit`, for example
`c.stim_circuit.get_detector_coordinates()`.

## Detector sampler or measurement sampler

`c.compile_detector_sampler()` samples detectors and observables;
`c.compile_sampler()` samples raw measurement records. They compile different problems, so
pick by what you need, not by convenience:

- The detector sampler builds a ZX diagram whose outputs are the detectors and observables
  themselves. After simplification, that diagram usually splits into a few small quantum
  components and many detectors that depend on the noise alone. Each of those is one parity
  of the noise and is sampled directly, with no state vector (`sampler.stats.num_direct_outputs`
  counts them). In a Clifford circuit every detector is direct.
- The measurement sampler must reproduce every random measurement outcome, so far fewer outputs
  are direct and more work goes through components.

When the task needs detectors or observables (decoding, postselection, logical error rates,
fidelity), use the detector sampler, even if the circuit is mostly Clifford. Measured on a
distance-5 surface code memory, it was about 10 times faster than the measurement sampler
(146 vs 14 million shots/s on 4 threads). Use the measurement sampler only when you need the
raw measurement bits. Do not sample measurements and compute detectors from them yourself.

## Sampling options

- `separate_observables=True` returns a `(detectors, observables)` tuple;
  `append_observables=True` returns one array with observables as the last columns.
- **Shots per call:** each `sample()` call has a fixed overhead. For fast circuits, ask for
  about 10^6 or more shots per call, limited by memory, instead of looping over small calls.
  Calls of 10^5 shots have been measured 2 to 4 times slower than calls of 10^7, and 100-shot
  calls about 80 times slower.
- **`batch_size`:** leave it at the default, which is sized from free memory and is close to
  optimal. Small values (a few thousand or less) are several times slower. Lower it only to
  cap memory; with the default strategy it does not change the samples.
- **`bit_packed=True`** returns little-endian packed `uint8` rows, as in Stim, using 8 times
  less memory. It is often faster for large circuits: on a distance-11 surface code memory
  (1,320 detectors) packed sampling was twice as fast as booleans, and at distance 5 about
  1.3 times as fast. For small circuits it gains little and is slower with
  `separate_observables` (the observable columns are split off and re-packed); use
  `append_observables=True` with packed output instead. Many decoders accept packed rows
  directly, for example `matching.decode_batch(dets, bit_packed_shots=True,
  bit_packed_predictions=True)` in PyMatching. Time both on your circuit when throughput
  matters.
- **Threads:** the default strategy runs on `numba.get_num_threads()` threads, all cores by
  default. Use `numba.set_num_threads(k)` before `sample` (or `NUMBA_NUM_THREADS=k` before
  importing Tsim) and leave cores free on a shared machine. Do not run several multi-threaded
  Tsim processes side by side; they oversubscribe the CPU.
- **Seeds:** a `seed` reproduces samples for the same Tsim and dependency versions and
  backend. Thread count and `batch_size` do not affect samples of the default strategy.

## Differences from Stim that change results

- Tsim returns the **raw values** of detectors and observables, and allows non-deterministic
  ones. Stim returns flips relative to a noiseless reference. For deterministic detectors, pass
  `use_detector_reference_sample=True` and `use_observable_reference_sample=True` to get Stim's
  convention; never use them for non-deterministic detectors. A noiseless circuit whose
  detectors are all 0 behaves the same either way.
- Tsim builds detector error models by handing Stim a Clifford stand-in of the circuit (`T`,
  `TPP` and `R_PAULI` become S-type rotations, `R_X`/`R_Y`/`R_Z`/`U3` become identity), so the
  model is only guaranteed correct for Clifford circuits and can be wrong for any detector or
  observable that a non-Clifford gate affects; treat it as an approximate decoder prior and
  judge decoders by their logical error rate on Tsim's own samples. For non-deterministic
  observables, pass `allow_gauge_detectors=True`; with coherent-noise rotations, see
  `tsim-qec-study`.

## Postselection

Many protocols discard every shot in which a check fails. Postselect inside Tsim whenever the
protocol does: it is often the largest speedup available, and choosing the wrong detectors
changes the result.

**How it works.** Pass a boolean mask of length `num_detectors` to
`CompiledDetectorSampler.sample`. A shot is discarded if any masked detector reads 1, and
**only surviving shots are returned**: asking for `shots=1_000_000` returns fewer rows, and
`len(dets) / shots` is the acceptance rate. Count requested and returned shots separately, and
keep calling until enough shots survive (see the loop in `tsim-qec-study`).

**Why it is fast.** With the default strategy, a shot stops being simulated at the first masked
detector that fires, and shots that a noise-only (direct) detector discards never reach the
state vector. The saving grows as the acceptance falls and as the discarding detectors come
earlier in the circuit. Distance-5 cultivation (14% acceptance) ran at 53.6 million shots/s
with postselection against 2.0 million without, on 4 threads. Filtering afterwards, which is
what `sinter` does, gives the same statistics but simulates every discarded shot in full.

**Which detectors to mask.** Follow the protocol's own definition, from the paper:

- Injection, cultivation and distillation postselect on every check, so for those circuits
  alone the mask usually covers every detector.
- When a cultivation circuit includes the escape stage (growing into a large surface code),
  mask only the injection and cultivation detectors. The escape-stage detectors go to a decoder,
  and shots are discarded afterwards by a decoder confidence threshold (the complementary gap),
  not by a mask. Masking every detector of an end-to-end circuit keeps almost no shots.
- Circuits sometimes end with noiseless checks that exist only to read out the result, such as
  a final ideal stabilizer measurement. Whether to mask them is part of the figure of merit, not
  of the protocol; decide it with the user and say which choice you made.

Detector coordinates or the circuit's structure usually identify the stages; if they do not,
ask the user rather than guessing.

```python
import numpy as np

mask = np.zeros(c.num_detectors, dtype=bool)
mask[0] = True
shots = 100_000
dets, obs = sampler.sample(shots=shots, postselection_mask=mask, separate_observables=True)
acceptance = len(dets) / shots
```

## Sampling strategy

Use the default strategy and leave `strategy=` unset. The other values (`"cat5"`, `"bss"`,
`"cutting"`) are provisional stabilizer-rank methods that are much slower and whose compile
memory grows exponentially with the number of T gates, so they can exhaust a machine's RAM in
seconds; do not use them, not even as a cross-check.

## Checking results

Cheap checks catch most modelling mistakes:

- **Clifford stand-in:** replace `T` by `S` and `T_DAG` by `S_DAG`, sample in Tsim and in
  Stim (with the reference-sample flags in Tsim), and compare detector rates within error bars.
- **Noiseless run:** in `c.without_noise()`, detectors meant to be deterministic must never fire.
- **Analytic case:** a T state under `DEPOLARIZE1(p)`, then noiseless `T_DAG`, `H`, `M`, gives
  `P(0) = 1 - 2p/3`.
- **Exact probabilities:** for a few noiseless qubits, `c.to_tensor()` gives the exact channel.

## Known limitations

- **Width can grow with circuit length** when non-Clifford rotations alternate with mid-circuit
  measurements on the same qubits: `k` repeats of `R_X(a) 0; MYY 0 1; R_Z(a) 1; MX 1` give
  width `k`. If the width grows linearly with the number of rounds, this is the cause; report it
  rather than trying to work around it.
- **Memory sizing ignores container limits.** The default batch size is chosen from the
  machine's free memory, not from a container or cgroup memory limit, so inside a memory-capped
  job pass an explicit `batch_size` (for example 2^16) to stay within the cap.
- **`str(circuit)` rounds noise probabilities** to 6 significant digits, as Stim does; rotation
  angles keep full precision. Keep the original file when exact noise strengths matter.

## Common errors

| Message | Fix |
|---|---|
| `strategy='active_statevector' needs N active qubits ... more than 30` | Width too large; see "Which circuits Tsim can simulate" |
| `... bytes per shot ... more than half of the available memory` | Width too large for this machine's RAM |
| `The circuit contains non-deterministic observables` | Pass `allow_gauge_detectors=True`, or fix the observable |
| `Malformed parametric tag` | Write `I[R_X(theta=0.25*pi)] 0` or `R_X(0.25) 0` |
| `'Circuit' object has no attribute ...` | A Stim method; call it on `c.stim_circuit` |
| `Cannot set NUMBA_NUM_THREADS ... once the threads have been launched` | Use `numba.set_num_threads(k)` |

## Reporting Tsim bugs

If Tsim crashes, returns statistics that contradict a check above, or is far slower than its
active width suggests, open an issue at https://github.com/QuEraComputing/tsim/issues. First
shrink the circuit until removing anything makes the problem disappear. Include the circuit
text, `tsim.__version__`, the strategy and seed, and the expected and actual results. Tell the
user before filing.
