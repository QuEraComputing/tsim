"""Benchmark strategy="statevector" on the clifft-bench circuits.

Run from the repository root:

    uv run python scripts/statevector/benchmark.py [--seconds S] [--seed N] [NAME ...]

Prefix the command with ``NPROC=1`` to run JAX on one thread: XLA sizes its CPU
thread pools from ``NPROC``.

The workloads are defined in ``workloads.py``. For every circuit this compiles
a detector sampler with ``strategy="statevector"``, makes one untimed warmup
call (JIT), then samples for at least ``--seconds`` with postselection on every
detector, counting as clifft-bench does: attempted shots, accepted shots (no
detector fired) and logical errors (observable 0 among accepted shots).
Acceptance is checked against Clifft 0.10.0's value (clifft-bench
release-v1-20260909-130000) within 5 standard errors. "cores busy" is the
process CPU time divided by wall time over the timed window.
"""

import argparse
import math
import time

import numpy as np
from workloads import CIRCUITS, WORKLOADS, cpu_seconds

import tsim


def count(sampler, shots: int, batch_size: int, num_detectors: int) -> tuple[int, int]:
    """Return (accepted shots, logical errors) of one sample call."""
    detectors, observables = sampler.sample(
        shots,
        batch_size=batch_size,
        separate_observables=True,
        postselection_mask=np.ones(num_detectors, dtype=bool),
    )
    keep = ~detectors.any(axis=1)
    return int(keep.sum()), int(observables[keep, 0].sum())


def main() -> None:
    """Benchmark the requested workloads and print one Markdown row each."""
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        "| circuit | compile s | warmup s | shots/s | cores busy | acceptance | "
        "Clifft acceptance | check | logical errors / accepted | sampler |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---|---:|---|")
    for name in args.names:
        file, reference, shots, batch_size = WORKLOADS[name]
        circuit = tsim.Circuit.from_file(str(CIRCUITS / file))
        start = time.perf_counter()
        sampler = circuit.compile_detector_sampler(
            strategy="statevector", seed=args.seed
        )
        compile_seconds = time.perf_counter() - start
        start = time.perf_counter()
        count(sampler, shots, batch_size, circuit.num_detectors)
        warmup_seconds = time.perf_counter() - start
        attempted = accepted = errors = 0
        cpu_start = cpu_seconds()
        start = time.perf_counter()
        while time.perf_counter() - start < args.seconds:
            kept, wrong = count(sampler, shots, batch_size, circuit.num_detectors)
            attempted += shots
            accepted += kept
            errors += wrong
        elapsed = time.perf_counter() - start
        cores = (cpu_seconds() - cpu_start) / elapsed
        acceptance = accepted / attempted
        sigma = math.sqrt(reference * (1 - reference) / attempted)
        check = "ok" if abs(acceptance - reference) <= 5 * sigma else "MISMATCH"
        print(
            f"| {name} | {compile_seconds:.2f} | {warmup_seconds:.2f} | "
            f"{attempted / elapsed:,.0f} | {cores:.1f} | {acceptance:.4f} | "
            f"{reference:.4f} | "
            f"{check} | {errors} / {accepted} | `{sampler!r}` |",
            flush=True,
        )


if __name__ == "__main__":
    main()
