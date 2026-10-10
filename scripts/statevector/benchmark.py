"""Benchmark strategy="statevector" on the clifft-bench circuits.

Run from the repository root:

    uv run python scripts/statevector/benchmark.py [--seconds S] [NAME ...]

Circuits are read from the directory in ``CLIFFT_BENCH_CIRCUITS`` (default
``/Users/rafaelhaenel/Documents/quera/clifft-bench/workloads/circuits``). For
every circuit this compiles a detector sampler with ``strategy="statevector"``,
makes one untimed warmup call (JIT), then samples for at least ``--seconds``
with postselection on every detector, counting as clifft-bench does: attempted
shots, accepted shots (no detector fired) and logical errors (observable 0
among accepted shots). Acceptance is checked against Clifft 0.10.0's value
(clifft-bench release-v1-20260909-130000) within 5 standard errors.
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

import tsim

CIRCUITS = Path(
    os.environ.get(
        "CLIFFT_BENCH_CIRCUITS",
        "/Users/rafaelhaenel/Documents/quera/clifft-bench/workloads/circuits",
    )
)

# name: (file, Clifft 0.10.0 acceptance, shots per sample call, batch size)
WORKLOADS = {
    "distillation": ("distillation.stim", 0.0798, 262_144, 65_536),
    "cultivation_d3": ("msc_d3_inject_cultivate_p1e-3.stim", 0.6868, 262_144, 65_536),
    "coherent_d3_r3": ("coherent_d3_r3.stim", 0.3732, 262_144, 65_536),
    "cultivation_d5": ("msc_d5_inject_cultivate_p1e-3.stim", 0.1440, 65_536, 16_384),
    "coherent_d5_r5": ("coherent_d5_r5.stim", 0.0078, 16_384, 4_096),
}


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
    args = parser.parse_args()
    print(
        "| circuit | compile s | warmup s | shots/s | acceptance | Clifft "
        "acceptance | check | logical errors / accepted | sampler |"
    )
    print("|---|---:|---:|---:|---:|---:|---|---:|---|")
    for name in args.names:
        file, reference, shots, batch_size = WORKLOADS[name]
        circuit = tsim.Circuit.from_file(str(CIRCUITS / file))
        start = time.perf_counter()
        sampler = circuit.compile_detector_sampler(strategy="statevector", seed=0)
        compile_seconds = time.perf_counter() - start
        start = time.perf_counter()
        count(sampler, shots, batch_size, circuit.num_detectors)
        warmup_seconds = time.perf_counter() - start
        attempted = accepted = errors = 0
        start = time.perf_counter()
        while time.perf_counter() - start < args.seconds:
            kept, wrong = count(sampler, shots, batch_size, circuit.num_detectors)
            attempted += shots
            accepted += kept
            errors += wrong
        elapsed = time.perf_counter() - start
        acceptance = accepted / attempted
        sigma = math.sqrt(reference * (1 - reference) / attempted)
        check = "ok" if abs(acceptance - reference) <= 5 * sigma else "MISMATCH"
        print(
            f"| {name} | {compile_seconds:.2f} | {warmup_seconds:.2f} | "
            f"{attempted / elapsed:,.0f} | {acceptance:.4f} | {reference:.4f} | "
            f"{check} | {errors} / {accepted} | `{sampler!r}` |",
            flush=True,
        )


if __name__ == "__main__":
    main()
