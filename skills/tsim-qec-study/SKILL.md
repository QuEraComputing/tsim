---
name: tsim-qec-study
description: Run quantum error correction studies with Tsim, such as logical error rate curves, postselected magic state cultivation or distillation, acceptance rates, and output-state fidelity, including decoder choice (PyMatching, Tesseract, BP-LSD, Relay-BP), shot budgets and error bars. Use this skill whenever someone wants a logical error rate, threshold plot, acceptance rate or fidelity from a circuit with T gates or rotations, wants to decode Tsim samples, or is about to use sinter with Tsim, even if they only say "simulate this protocol" or "reproduce this figure".
license: Apache-2.0
compatibility: Python 3.11+, bloqade-tsim (main). Decoders are separate packages.
metadata:
  author: QuEra Computing
  version: "1.1"
---

# QEC studies with Tsim

This skill builds on the `tsim` skill (circuit format, active width, sampling options); read it
first if it is available. The workflow below turns a noisy circuit into numbers that can go in a
paper: a logical error rate or infidelity per accepted shot, an acceptance rate, error bars, and
a script that regenerates them.

## Workflow

1. **Build the circuit** for each parameter point (noise strength, distance, rounds). Keep it in
   a function so the final script can regenerate everything.
2. **Compile and check the cost**: read `sampler.active_width`, then time a warm-up call and
   one call of about 10^6 shots. Estimate the time for the full budget (sampling and
   decoding) before launching anything long, and tell the user.
3. **Decide what is postselected and what is decoded.** Postselection detectors go into a
   `postselection_mask`. Detectors that feed a decoder stay unmasked.
4. **Pick a decoder** (or none) from the table below.
5. **Sample in large calls, decode, and accumulate counts**, using the loop below.
6. **Report** each point with shots, kept shots, errors, acceptance, logical error rate per
   kept shot and its confidence interval, and seconds spent sampling and decoding.

## Do not use `sinter.collect` with Tsim

sinter can call Tsim, but it costs speed and can cost correctness:

- sinter applies `postselection_mask` after sampling, so Tsim simulates every discarded shot in
  full. Postselecting inside Tsim stops discarded shots early, which for cultivation-style
  protocols is often the largest speedup available.
- sinter reads raw detector values. Tsim returns raw values, not flips relative to a noiseless
  reference, so a detector that is 1 without noise looks like an error to the decoder.
- Each sinter worker process compiles its own sampler and starts its own numba thread pool, so
  several workers oversubscribe the CPU.

Write the short loop below instead. sinter's helpers are still useful for the statistics
(`sinter.fit_binomial`) and for plotting.

## The sampling loop

```python
import numpy as np
import tsim


def run_point(circuit, *, shots, postselection_mask=None, decode=None,
              shots_per_call=1_000_000, seed=None):
    """Return counts for one parameter point."""
    sampler = circuit.compile_detector_sampler(seed=seed)
    kept = errors = 0
    remaining = shots
    while remaining > 0:
        n = min(shots_per_call, remaining)
        dets, obs = sampler.sample(
            shots=n, postselection_mask=postselection_mask, separate_observables=True
        )
        predicted = decode(dets) if decode is not None else np.zeros_like(obs)
        errors += int(np.any(predicted != obs, axis=1).sum())
        kept += len(dets)
        remaining -= n
    return {"shots": shots, "kept": kept, "errors": errors}
```

Notes:

- One sampler is compiled per point and reused across calls; successive calls continue the
  random stream, so seeds never repeat. Use a different `seed` per point.
- Always use the detector sampler here, not the measurement sampler: it samples most
  detectors directly from the noise and is often an order of magnitude faster.
- Keep `shots_per_call` large (10^6 or more, limited by memory) because each call has a fixed
  overhead. Leave `batch_size` at its default.
- For circuits with hundreds of detectors or more, sample with `bit_packed=True` and
  `append_observables=True`, and give the decoder packed rows (PyMatching:
  `decode_batch(dets, bit_packed_shots=True, bit_packed_predictions=True)`). This uses 8 times
  less memory and was twice as fast at 1,320 detectors. Unpack observables with
  `np.unpackbits(..., bitorder="little")`.
- An error is any observable that differs from its prediction. If observables are not
  deterministic without noise (for example a magic state measured directly), a logical error
  rate is not meaningful; measure fidelity instead (see below).
- Detectors that are 1 in the noiseless circuit break both postselection and decoding. Check
  with `c.without_noise()` that the detectors you mask or decode are 0 without noise; if a
  detector is deterministic but equals 1, fix its definition or use
  `use_detector_reference_sample=True`.

## Choosing a decoder

Build the decoder from `c.detector_error_model(...)`. Tsim builds detector error models by
handing Stim a Clifford stand-in of the circuit (`T`, `TPP` and `R_PAULI` become S-type
rotations, `R_X`/`R_Y`/`R_Z`/`U3` become identity), so the model is only guaranteed correct
for Clifford circuits and can be wrong for any detector or observable that a non-Clifford gate
affects; treat it as an approximate decoder prior and judge decoders by their logical error
rate on Tsim's own samples. For non-deterministic observables, pass
`allow_gauge_detectors=True`.

When rotations model noise (coherent over-rotations), the plain model leaves them out and the
decoder never learns about them. Give the decoder their Pauli twirl instead: replace each
`R_P(a)` by `P_ERROR(sin²(aπ/2))`, then build the model from that circuit. Keep sampling the
original circuit; only the decoder's model changes. Rotations that are part of the protocol,
such as T gates or a deliberate rotation, are not noise; leave them alone.

```python
import math
import re

_ROTATION = re.compile(r"^(\s*)R_([XYZ])\(([^)]*)\)", re.MULTILINE)


def twirled(circuit):
    """Copy of ``circuit`` with each R_X/R_Y/R_Z(a) replaced by its Pauli twirl."""

    def repl(m):
        p = math.sin(float(m.group(3)) * math.pi / 2) ** 2
        return f"{m.group(1)}{m.group(2)}_ERROR({p:.12g})"

    return tsim.Circuit(_ROTATION.sub(repl, str(circuit)))


noisy = tsim.Circuit("R 0\nR_X(0.02) 0\nX_ERROR(0.001) 0\nM 0\nDETECTOR rec[-1]")
dem = twirled(noisy).detector_error_model(decompose_errors=True)
```

The regex handles `R_X`, `R_Y` and `R_Z` written one per line, as `str(circuit)` prints them;
extend it if the noise uses `R_PAULI` or `U3`.

| Decoder | Package | Use when | Notes |
|---|---|---|---|
| PyMatching | `pymatching` | Surface and repetition codes whose errors flip at most two detectors | Fast. Needs `decompose_errors=True`; it silently drops hyperedges otherwise. Fails on models that cannot be decomposed (color codes, many LDPC codes). |
| Tesseract | `tesseract-decoder` | Accuracy matters most; small and medium codes; any error model | Search decoder, close to most-likely-error. Can be very slow on large or noisy models; time it first. |
| BP-LSD | `ldpc` | qLDPC and color codes, hyperedge models | Fast BP with localized statistics decoding; good default for LDPC codes. |
| Relay-BP | `relay-bp[stim]` | qLDPC codes (bivariate bicycle) and large surface codes | BP ensemble from IBM; fast and accurate on BB codes. |

Follow the paper being reproduced when it names a decoder. Otherwise start with PyMatching for
matchable codes, BP-LSD or Relay-BP for LDPC codes, and Tesseract when accuracy matters and the
model is small enough. Decoding can take longer than sampling; time both separately and report
both. Tested code for each decoder is in `references/decoders.md`.

## Postselected protocols and fidelity

For cultivation, distillation and injection, the figure of merit is usually the infidelity of
the accepted output state, together with the acceptance rate. These protocols often need no
decoder at all: postselect on every check, then measure the output.

To measure the fidelity of an output magic state without special instructions, append a
noiseless inverse of the encoding that moves the logical state onto one qubit, then rotate the
target state to |0⟩ and measure. For a T state |T⟩ = T|+⟩, the rotation is `T_DAG` then `H`;
the probability of outcome 0 is exactly ⟨T|ρ|T⟩. Make that measurement an observable, so the
observable's flip rate over kept shots is the infidelity.

```python
p = 0.01
c = tsim.Circuit(
    f"""
    RX 0
    T 0
    DEPOLARIZE1({p}) 0
    T_DAG 0
    H 0
    M 0
    OBSERVABLE_INCLUDE(0) rec[-1]
    """
)
counts = run_point(c, shots=200_000, seed=1)
infidelity = counts["errors"] / counts["kept"]  # close to 2p/3
```

Everything after the noisy protocol (unencoding, `T_DAG`, `H`, `M`) must be noiseless,
because it is a measurement device, not part of the protocol.

## Error bars and shot budgets

- The logical error rate per kept shot is `errors / kept`; its relative error is about
  `1 / sqrt(errors)`. Aim for at least 100 errors per point for a 10% error bar.
- For confidence intervals use `sinter.fit_binomial(num_shots=kept, num_hits=errors,
  max_likelihood_factor=1000)`, or a Wilson interval. Report the acceptance rate
  `kept / shots` with its own interval.
- Plan the budget from measured rates: shots needed ≈ target errors / (acceptance × logical
  error rate). If that exceeds the time available, say so and propose fewer points, a smaller
  instance, or a higher noise strength.

## Reproducibility

Put the whole study in one script that builds every circuit, runs every point, writes raw
counts (shots, kept, errors, seconds) to JSON or CSV, and draws the figures from that file. Use
fixed, distinct seeds per point and record `tsim.__version__`. When adding shots later, run new
seeds and add counts; do not reuse a seed.
