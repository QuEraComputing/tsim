---
name: simulating-circuits-with-tsim
description: "Samples noisy quantum circuits with non-Clifford gates using Tsim (bloqade-tsim), a Stim-compatible simulator, and runs Monte Carlo QEC studies with it. Use when a task involves Tsim or bloqade-tsim, or a noisy Stim-format circuit with T, T_DAG, R_X, R_Y, R_Z, U3, R_PAULI, TPP, CCZ, CCX or CS gates (magic state cultivation, distillation, injection, coherent rotation errors), even if Tsim is not named: judging feasibility or run time (active width, k_max); sampling detectors, measurements or observables; postselection; logical error rates, thresholds, acceptance or output infidelity with error bars; decoding Tsim samples (PyMatching, Tesseract, BP-LSD, Relay-BP); detector error models; sinter with Tsim; or porting published circuits (arXiv, GitHub, Zenodo) whose T or CCZ gates are Clifford stand-ins. Also use for errors such as \"needs N active qubits\", \"Malformed parametric tag\" or \"non-deterministic observables\", detectors that read 1 where Stim reports 0, or slow sample() calls."
license: Apache-2.0
compatibility: Python 3.11+ and Tsim 1.1.0 or later (bloqade-tsim). Decoders are separate packages.
metadata:
  author: QuEra Computing
  version: "2.0"
  last-reviewed: "2026-11-10"
  tested-with: "Tsim 1.1.0"
---

# Tsim

Tsim samples noisy quantum circuits with non-Clifford gates. It reads Stim's circuit format plus
a few non-Clifford instructions and mirrors Stim's sampler API, so most Stim code ports by
changing `stim` to `tsim`. Install with `uv add bloqade-tsim` or `pip install bloqade-tsim`
(1.1.0 or later); import as `tsim`. Docs: https://bloqade.quera.com/tsim (append `.md` to any
page for Markdown; an MCP server for searching them is at https://bloqade.quera.com/tsim/mcp).

This file covers what every task needs. Open the references when the task calls for them:

- [references/qec-experiments.md](references/qec-experiments.md): Monte Carlo estimates of
  logical error rates, acceptance and fidelity; the sampling loop; decoders and their error
  models; error bars; why not to use `sinter.collect`.
- [references/decoders.md](references/decoders.md): tested code for PyMatching, Tesseract,
  BP-LSD and Relay-BP on Tsim samples.
- [references/porting.md](references/porting.md): finding a paper's circuits, converting
  Clifford stand-ins to real T or CCZ gates, and validating the conversion.

## Common mistakes

Check these before writing code; each links to the details below.

1. Comparing Tsim's detector values with Stim's without the reference flags; Tsim returns raw
   values ("Differences from Stim").
2. Sampling measurements and computing detectors from them instead of using the detector
   sampler ("Detector sampler or measurement sampler").
3. Postselecting after sampling, or with `sinter.collect`, instead of passing
   `postselection_mask` ("Postselection").
4. Using `strategy="cat5"`, `"bss"` or `"cutting"`, when not explicitly asked by the user ("Sampling strategy").
5. Writing angles in radians: `R_Z(0.25)` is a rotation by π/4 ("Circuit format").
6. Many small `sample()` calls or a small `batch_size` ("Sampling options").

## Which circuits Tsim can simulate

Whether Tsim can simulate a circuit is decided by the **active width** w: the largest number
of qubits that must be held in a state vector at once, after Tsim has removed everything
Clifford it can. The state-vector work per shot grows as 2^w. The number of T gates and the
number of qubits do not set it. End-to-end magic state cultivation at distance 5 (463 qubits,
53 T gates) has width 6, a distance-5 surface code with coherent rotations on every qubit (64
qubits, over 1,000 rotations) has width 11, and a 20-qubit quantum volume circuit has width 20.

The width decides feasibility, not speed. Below a width of about 10 the state vector is cheap,
and the speed is set by the number of noise channels and detectors instead: published circuits
of width 1 run anywhere from about 1 to 100 million shots per second on one core.

Why the width is often small: ZX simplification absorbs the Clifford gates, so only the
non-Clifford rotations and the measurements that interact with them reach the state vector.
Adding Clifford structure around a non-Clifford core leaves the width unchanged and moves the
cost elsewhere. Distance-5 cultivation has width 6 both as a 42-qubit inject-and-cultivate
circuit and as a 463-qubit end-to-end circuit with a surface-code escape stage. The end-to-end
circuit has 2,744 detectors, 2,730 of them sampled directly from the noise, and once
postselection has removed most of the state-vector work, that direct sampling is the bottleneck
(3.3 vs 53.6 million shots/s for the smaller circuit on 4 threads).

Compile first, then read the width. Compilation takes seconds for circuits with a few thousand
detectors and about a minute at 50,000; sampling is usually where time goes.

```python
import tsim

c = tsim.Circuit(
    """
    R 0 1
    H 0
    T 0
    CNOT 0 1
    DEPOLARIZE2(0.01) 0 1
    CNOT 0 1
    T_DAG 0
    H 0
    M 0 1
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-2]
    """
)
sampler = c.compile_detector_sampler(seed=1)
assert sampler.active_width == 1  # 0 would mean no state vector is needed
print(sampler.stats)  # component widths, peak bytes per shot, ...
dets, obs = sampler.sample(shots=100_000, separate_observables=True)
```

| Active width | What to expect on a typical multi-core CPU |
|---|---|
| 0 | No state vector; as fast as sampling the noise |
| 1 to about 16 | Comfortable; millions of shots are practical |
| about 17 to 26 | Slow; each extra qubit doubles the cost, and near 26 a shot takes seconds |
| 27 to 30 | Accepted, but each shot needs gigabytes and takes minutes; rarely practical |
| above 30 | Rejected with a `ValueError` (or earlier, if memory runs out) |

If `sampler.active_width` raises `AttributeError`, the installed Tsim is older than 1.1.0;
upgrade `bloqade-tsim` before going further.

Before a long run, time it: one small warm-up call (the first call compiles kernels), then
one call of about 10^6 shots, and extrapolate. Small calls understate the rate, on a GPU even
more. Report the width, the measured shots per second and where it ran (`sampler.stats.device`,
and the thread count on the CPU), along with any plan.

If the width is too large: postselect inside Tsim (discarded shots stop early, see below),
simulate a smaller instance (lower distance or fewer rounds) and report how the width grows,
drop observables the study does not need (see qec-experiments.md), or check whether part of
the circuit can be a Clifford stand-in.

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
`S[T]`, `T_DAG` is `S_DAG[T]`, and `R_Z(0.25)` is `I[R_Z(0.25)]` or, in some versions,
`I[R_Z(theta=0.25*pi)]`; Tsim reads both. `c.stim_circuit` returns that `stim.Circuit`. Stim
reads it as a Clifford stand-in (T as S, rotations as identity). Many `stim.Circuit` methods
are not on `tsim.Circuit`; call them on `c.stim_circuit`, for example
`c.stim_circuit.get_detector_coordinates()`.

## Differences from Stim that change results

- **Raw detector values.** Stim reports each detector and observable as a flip relative to a
  noiseless reference sample; Tsim reports its raw value, and allows random ones. A detector
  whose noiseless value is 1 reads 1 in every Tsim shot and 0 in Stim, which changes detector
  rates, postselection and decoding. To get Stim's convention, sample with
  `use_detector_reference_sample=True` and `use_observable_reference_sample=True`, but only
  when the detectors and observables are deterministic without noise; for random ones (for
  example a measured magic state) the reference is meaningless.
- **Detector error models.** `c.detector_error_model()` is built by Stim from the Clifford
  stand-in, so it is only guaranteed correct for Clifford circuits. Before using it as a
  decoder prior, read "Choosing a decoder" in qec-experiments.md.

## Detector sampler or measurement sampler

`c.compile_detector_sampler()` samples detectors and observables; `c.compile_sampler()` samples
raw measurement records. They compile different problems:

- The detector sampler builds a ZX diagram whose outputs are the detectors and observables
  themselves. After simplification it usually splits into a few small quantum components and
  many detectors that depend on the noise alone, each sampled directly as one parity of the
  noise (`sampler.stats.num_direct_outputs` counts them). In a Clifford circuit every detector
  is direct.
- The measurement sampler must reproduce every random measurement outcome, so far fewer outputs
  are direct.

When the task needs detectors or observables (decoding, postselection, logical error rates,
fidelity), use the detector sampler, even if the circuit is mostly Clifford: on a distance-5
surface code memory it was about 10 times faster (146 vs 14 million shots/s on 4 threads). Use
the measurement sampler only for raw measurement bits; do not compute detectors from them
yourself.

## Sampling options

- `separate_observables=True` returns a `(detectors, observables)` tuple;
  `append_observables=True` returns one array with observables as the last columns.
- **Shots per call:** each `sample()` call has a fixed overhead. For fast circuits, ask for
  about 10^6 or more shots per call, limited by memory, instead of looping over small calls.
  Calls of 10^5 shots have been measured 2 to 4 times slower than calls of 10^7, and 100-shot
  calls about 80 times slower. On a GPU the fixed cost per call is larger (the call waits for
  the GPU and copies its rows back), so ask for 10^6 to 10^7 shots per call there.
- **`batch_size`:** leave it at the default, which is sized from free memory and is close to
  optimal. Small values (a few thousand or less) are several times slower. Lower it only to
  cap memory; on the CPU it does not change the samples. On a GPU the default batch is 2^20
  shots, one GPU launch, all launched back to back; `batch_size` can only lower it, a small
  one makes launch overhead dominate, and it changes which samples a seed gives.
- **`bit_packed=True`** returns little-endian packed `uint8` rows, as in Stim, using 8 times
  less memory. On a distance-11 surface code memory (1,320 detectors) it was twice as fast as
  booleans, at distance 5 about 1.3 times. On an Apple GPU the gap is larger, since booleans
  move 8 times more bytes off the GPU: 2.3 to 3.2 times on circuits with hundreds of
  detectors (surface_d7_r7: 757M against 237M shots/s on an M4 Pro). Use packed rows for
  large studies and decode them packed. For small circuits it gains little and is slower
  with `separate_observables` (the observables are split off and re-packed); use
  `append_observables=True` with packed output.
- **Threads:** the default strategy runs on `numba.get_num_threads()` threads, all cores by
  default. Use `numba.set_num_threads(k)` before `sample` (or `NUMBA_NUM_THREADS=k` before
  importing Tsim) and leave cores free on a shared machine. Do not run several multi-threaded
  Tsim processes side by side; they oversubscribe the CPU.
- **Apple GPUs:** on an Apple-silicon Mac the default strategy samples on the GPU when every
  component has at most 11 active qubits, often 5 to 10 times faster than all CPU cores. That
  includes Clifford circuits, whose detectors come from the noise alone: the GPU samples the
  noise too. It
  computes outcome probabilities in float32 and never draws an outcome below about 1e-11
  (circuits with rotations by less than about 1e-4 π stay on the CPU). Thread settings do not
  affect GPU sampling.
- **`device`** (in `compile_sampler` and `compile_detector_sampler`) chooses where it samples:
  `"auto"` (default) uses the GPU when it can and otherwise the CPU, silently; `"cpu"` stays
  on the CPU in double precision; `"metal"` raises with the reason if the GPU cannot run the
  circuit (no Apple GPU, width above 11, a tiny rotation, out of GPU memory). Check
  `sampler.stats.device` (or the repr, `samples on metal`) after compiling, since `"auto"`
  does not warn. Pass `device="metal"` for long runs that must not drop to the CPU unnoticed,
  and `device="cpu"` to reproduce samples across machines or to compare against the GPU.
  `TSIM_DEVICE=cpu` in the environment keeps `"auto"` on the CPU.
- **Processes:** sample from one process per GPU. Threads sharing a GPU sampler run its calls
  one at a time, and several processes compete for the same GPU. A process forked after its parent used the GPU cannot use
  it (it samples on the CPU, or raises with `device="metal"`); start workers with the
  `"spawn"` or `"forkserver"` method (macOS's default is `"spawn"`).
- **Seeds:** a `seed` reproduces samples for the same Tsim and dependency versions and
  device. On the CPU, thread count and `batch_size` do not affect the samples; on a GPU,
  `batch_size` does, and GPU samples differ from CPU samples, so with `device="auto"` a seed
  reproduces only on the same kind of machine.

## Postselection

Many protocols discard every shot in which a check fails. Postselect inside Tsim whenever the
protocol does: it is often the largest speedup available, and choosing the wrong detectors
changes the result.

**How it works.** Pass a boolean mask of length `num_detectors` to
`CompiledDetectorSampler.sample`. A shot is discarded if any masked detector reads 1, and
**only surviving shots are returned**: asking for `shots=1_000_000` returns fewer rows, and
`len(dets) / shots` is the acceptance rate. Count requested and returned shots separately, and
keep calling until enough shots survive (see the loop in qec-experiments.md).

**Why it is fast.** A shot stops being simulated at the first masked detector that fires, and
shots that a noise-only (direct) detector discards never reach the state vector. The saving
grows as the acceptance falls and as the discarding detectors come earlier in the circuit.
Distance-5 cultivation (14% acceptance) ran at 53.6 million shots/s with postselection against
2.0 million without, on 4 threads. Filtering afterwards, which is what `sinter` does, gives the
same statistics but simulates every discarded shot in full.

**Which detectors to mask.** Follow the protocol's own definition, from the paper, and use the
circuit's own marking of postselected detectors (a tag, a coordinate, a comment) when it has
one:

- Injection, cultivation and distillation postselect on every check, so for those circuits
  alone the mask usually covers every detector.
- When a cultivation circuit includes the escape stage (growing into a large surface code),
  mask only the injection and cultivation detectors. The escape-stage detectors go to a decoder,
  and shots are discarded afterwards by a decoder confidence threshold (the complementary gap),
  not by a mask. Masking every detector of an end-to-end circuit keeps almost no shots. This
  skill does not cover computing the gap; use the paper's code for it, or ask the user.
- Circuits sometimes end with noiseless checks that exist only to read out the result, such as
  a final ideal stabilizer measurement. Whether to mask them is part of the figure of merit, not
  of the protocol; decide it with the user and say which choice you made.
- Postselecting a stage that the paper decodes is a different protocol. If you do it anyway,
  for example to avoid writing a decoder, label the result as such.

Detector coordinates (`c.stim_circuit.get_detector_coordinates()`) or the circuit's structure
usually identify the stages; if they do not, ask the user rather than guessing.

Some detectors are random by design for some parameter values, for example in a sweep of the
injected angle. Do not use the reference flags for those; postselect on their raw value, and
run the noiseless check at a parameter value where they are deterministic (the T angle). The
two reference flags are independent: a circuit can need `use_detector_reference_sample=True`
with `use_observable_reference_sample=False`.

**Check the mask before sampling.** A wrong mask gives plausible but wrong numbers. List the
stages down, and check that no masked detector fires without noise: one that is always 1 or
random discards shots that have no error. Sample the noiseless check with the same reference
flags as the real run. `without_noise()` removes the noise channels but keeps every rotation,
so if rotations model coherent noise, delete those lines too for this check, and keep the
rotations that belong to the protocol.

```python
import numpy as np

# Detector indices of each postselected stage, taken from the protocol.
stages = {"checks": [0, 1]}
mask = np.zeros(c.num_detectors, dtype=bool)
for indices in stages.values():
    mask[indices] = True

quiet = c.without_noise().compile_detector_sampler(seed=0).sample(shots=1000)
fired = np.flatnonzero(mask & quiet.any(axis=0))
assert not fired.size, f"masked detectors {fired} fire without noise"

shots = 100_000
dets, obs = sampler.sample(shots=shots, postselection_mask=mask, separate_observables=True)
acceptance = len(dets) / shots
```

To see which stage discards most shots, sample once without a mask and compute
`1 - dets[:, stages[name]].any(axis=1).mean()` per stage. Report the stages and their detectors
with the results.

## Sampling strategy

Use the default strategy and leave `strategy=` unset. The other values (`"cat5"`, `"bss"`,
`"cutting"`) are provisional stabilizer-rank methods that are much slower and whose compile
memory grows exponentially with the number of T gates, so they can exhaust a machine's RAM in
seconds. Do not use them, not even as a cross-check, unless explicitly asked by the user.

## Checking results

Cheap checks catch most modelling mistakes. Before reporting numbers, copy this checklist and
tick it off:

```
Tsim results:
- [ ] Active width read; rate timed with a warm-up and a large call; device and threads noted
- [ ] Noiseless run: detectors used for postselection or decoding never fire
- [ ] Reference flags used only if detectors and observables are deterministic
- [ ] At least one check below passed (stand-in vs Stim, to_tensor, independent simulator)
```

- **Clifford stand-in:** replace `T` by `S` and `T_DAG` by `S_DAG`, sample in Tsim (with both
  reference flags) and in Stim, and compare detector rates within error bars.
- **Noiseless run:** in `c.without_noise()` (also without rotations that model noise),
  detectors meant to be deterministic must never fire.
- **Exact probabilities:** for a few noiseless qubits, `c.to_tensor()` gives the exact channel.
- **Independent result for the real circuit:** the checks above do not test a non-Clifford
  result at scale. When it matters, or when it disagrees with a paper, compare one or two
  points with an independent near-Clifford simulator (for example Clifft) or with exact values
  from the literature.

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
- **`CCZ`, `CCX`, `CS` and `CS_DAG` are stored as their T-gate decompositions**, so `str(c)`
  prints the decomposition and `c.tcount()` counts 7 per `CCZ` or `CCX`. Convert before
  comparing with a paper's gate count.
- **`R_PAULI` acts on at most 64 qubits per Pauli product**; longer products raise a
  `ValueError` at compile time.

## Common errors

| Message | Fix |
|---|---|
| `strategy='active_statevector' needs N active qubits ... more than 30` | Width too large; see "Which circuits Tsim can simulate" |
| `... bytes per shot ... more than half of the available memory` | Width too large for this machine's RAM |
| `non-deterministic observables` or `The number of observables changed` (from `detector_error_model`) | An observable is random without noise; no flag fixes this. Make it deterministic (for a magic state, see the fidelity measurement in qec-experiments.md) |
| `non-deterministic detectors` (from `detector_error_model(decompose_errors=True)`) | Pass `allow_gauge_detectors=True` as well |
| `Malformed parametric tag` | Angles are in units of π: write `R_X(0.25) 0` or `I[R_X(0.25)] 0` |
| `'Circuit' object has no attribute ...` | A Stim method; call it on `c.stim_circuit` |
| `Cannot set NUMBA_NUM_THREADS ... once the threads have been launched` | Use `numba.set_num_threads(k)` |

## Reporting bugs and circuits Tsim cannot simulate yet

Issues go to https://github.com/QuEraComputing/tsim/issues. Issues are public, so tell the user
before filing and ask before sharing a circuit that is not published. Search first
(`gh issue list --repo QuEraComputing/tsim --state all --search "<keywords>"`), and add to an
existing issue rather than opening a duplicate. Do not patch or monkey-patch Tsim inside the
user's project to get around a bug; report it, and offer the user a workaround to choose.

**Bugs.** If Tsim crashes, returns statistics that contradict a check above, or is far slower
than its active width suggests, first shrink the circuit until removing anything makes the
problem disappear. Include the circuit text, `tsim.__version__` and where Tsim was installed
from (PyPI, or git and the commit), the seed, and the expected and actual results.

**Circuits that are out of reach.** If a circuit matters for the user's work but Tsim cannot
simulate it in practice (the width is over 30, or shots are too slow for the study), suggest
opening an issue. The maintainers want these cases: they show where Tsim needs to improve, and
a real circuit is the best benchmark for that. Here, do not shrink the circuit; the full circuit
is the point, but only if the user is open to share it.  Include the circuit
(attach the file, or link to it if it is large), where it
comes from (paper or repository), what the user wants to compute, and what you measured:
qubits, T count (`c.tcount()`), `sampler.active_width` and `sampler.stats` (or the error
message), and the time per shot if sampling ran at all.
