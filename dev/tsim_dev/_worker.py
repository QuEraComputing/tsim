"""Measure one circuit in a fresh process and print one JSON line.

This file is run by path with the Python of the checkout under test, which may be
an older Tsim, so it uses only the standard library, numpy and Tsim's public API.
"""

from __future__ import annotations

import argparse
import json
import time


def phase_import():
    """Import Tsim (a named frame, so profiles can tell the phases apart)."""
    import tsim

    return tsim


def phase_compile(tsim, args):
    """Parse the circuit and compile the sampler."""
    circuit = tsim.Circuit.from_file(args.circuit)
    kind = args.sampler
    if kind == "auto":
        has_outputs = circuit.num_detectors + circuit.num_observables > 0
        kind = "detector" if has_outputs else "measurement"
    kwargs = {"seed": args.seed}
    if args.strategy:
        kwargs["strategy"] = args.strategy
    if kind == "detector":
        sampler = circuit.compile_detector_sampler(**kwargs)
    else:
        sampler = circuit.compile_sampler(**kwargs)
    return circuit, kind, sampler


def make_sample(circuit, kind, sampler, args):
    """Return a function that samples a number of shots with the requested options."""
    import numpy as np

    kwargs = {}
    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
    if args.bit_packed:
        kwargs["bit_packed"] = True
    if args.postselect == "all":
        if kind != "detector":
            raise ValueError("postselection needs the detector sampler")
        kwargs["postselection_mask"] = np.ones(circuit.num_detectors, dtype=np.bool_)

    def sample(shots):
        return sampler.sample(shots, **kwargs)

    return sample


def phase_sample(sample, shots, seconds):
    """Sample repeatedly for about `seconds` (the phase that profiles look at)."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        sample(shots)


def calibrate(sample, min_seconds, max_shots):
    """Find a shot count and a number of calls that together take `min_seconds`.

    The shot count doubles until one call is long enough. It stops at `max_shots`,
    which bounds the size of the returned array; faster circuits are timed over
    several calls instead.
    """
    shots = 16
    while True:
        start = time.perf_counter()
        sample(shots)
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds:
            return shots, 1
        if shots >= max_shots:
            return shots, max(1, round(min_seconds / max(elapsed, 1e-6)))
        grow = min(16.0, max(2.0, 1.2 * min_seconds / max(elapsed, 1e-6)))
        shots = min(max_shots, int(shots * grow))


def main():
    """Run the requested mode and print `RESULT <json>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--mode", default="bench")  # bench, prime, inspect, profile
    parser.add_argument("--sampler", default="auto")
    parser.add_argument("--postselect", default="none")
    parser.add_argument("--strategy", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--bit-packed", action="store_true")
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    start = time.perf_counter()
    tsim = phase_import()
    imported = time.perf_counter()
    circuit, kind, sampler = phase_compile(tsim, args)
    compiled = time.perf_counter()

    result = {
        "tsim_version": getattr(tsim, "__version__", "?"),
        "sampler": kind,
        "import_s": imported - start,
        "compile_s": compiled - imported,
        "active_width": getattr(sampler, "active_width", None),
        "qubits": circuit.num_qubits,
        "detectors": circuit.num_detectors,
        "measurements": circuit.num_measurements,
        "tcount": circuit.tcount(),
    }
    if args.mode == "inspect":
        result["stats"] = str(getattr(sampler, "stats", ""))
        print("RESULT " + json.dumps(result))
        return

    sample = make_sample(circuit, kind, sampler, args)
    warm = time.perf_counter()
    sample(16)
    result["warmup_s"] = time.perf_counter() - warm
    if args.mode == "prime":
        print("RESULT " + json.dumps(result))
        return

    outputs = circuit.num_detectors + circuit.num_observables
    if kind == "measurement":
        outputs = circuit.num_measurements
    # NOTE: one call returns at most about 256 MB of unpacked samples.
    shots, calls = calibrate(sample, args.min_seconds, (1 << 28) // max(1, outputs))
    if args.mode == "profile":
        phase_sample(sample, shots, args.min_seconds * 10)
        result["shots"] = shots
        print("RESULT " + json.dumps(result))
        return

    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        for _ in range(calls):
            sample(shots)
        times.append(time.perf_counter() - start)
    total = shots * calls
    result.update(shots=total, seconds=min(times), shots_per_s=total / min(times))
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
