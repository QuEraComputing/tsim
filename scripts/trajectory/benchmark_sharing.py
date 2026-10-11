"""Benchmark trajectory sharing on the clifft-bench circuits.

Run from the repository root:

    uv run python scripts/trajectory/benchmark_sharing.py [--seconds S] [NAME ...]

Same counting as ``scripts/statevector/benchmark.py`` (postselection on every
detector, acceptance checked against Clifft 0.10.0 within 5 standard errors,
logical errors = observable 0 among accepted shots), for three samplers per
workload: one lane per shot (``trajectory_sharing=False``), shared trajectories
at the benchmark's batch size, and shared trajectories at a larger batch size,
which sharing allows because a batch no longer needs one state vector per shot.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import tsim

sys.path.insert(0, str(Path(__file__).parents[1] / "statevector"))
from benchmark import CIRCUITS, WORKLOADS

SHARED_BATCH = 65_536


def count(sampler, shots, batch_size, num_detectors):
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
    """Benchmark the requested workloads and print one Markdown row per sampler."""
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    print(
        "| circuit | sampler | batch | warmup s | shots/s | acceptance | Clifft "
        "acceptance | check | logical errors / accepted | mean groups | mean peak lanes |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|")
    for name in args.names:
        file, reference, shots, batch_size = WORKLOADS[name]
        circuit = tsim.Circuit.from_file(str(CIRCUITS / file))
        configurations = [
            ("one lane per shot", False, batch_size, shots),
            ("shared", True, batch_size, shots),
            ("shared", True, SHARED_BATCH, max(shots, 4 * SHARED_BATCH)),
        ]
        for label, sharing, batch, per_call in configurations:
            sampler = circuit.compile_detector_sampler(
                strategy="statevector", seed=0, trajectory_sharing=sharing
            )
            start = time.perf_counter()
            count(sampler, per_call, batch, circuit.num_detectors)
            warmup = time.perf_counter() - start
            sampler.sharing_stats.clear()
            attempted = accepted = errors = 0
            start = time.perf_counter()
            while time.perf_counter() - start < args.seconds:
                kept, wrong = count(sampler, per_call, batch, circuit.num_detectors)
                attempted += per_call
                accepted += kept
                errors += wrong
            elapsed = time.perf_counter() - start
            acceptance = accepted / attempted
            sigma = math.sqrt(reference * (1 - reference) / attempted)
            check = "ok" if abs(acceptance - reference) <= 5 * sigma else "MISMATCH"
            stats = sampler.sharing_stats
            groups = np.mean([s["groups"] for s in stats]) if stats else float("nan")
            peak = np.mean([s["peak lanes"] for s in stats]) if stats else float("nan")
            print(
                f"| {name} | {label} | {batch:,} | {warmup:.2f} | "
                f"{attempted / elapsed:,.0f} | {acceptance:.4f} | {reference:.4f} | "
                f"{check} | {errors} / {accepted} | {groups:.0f} | {peak:.0f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
