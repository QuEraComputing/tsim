"""Time Tsim's statevector strategy on the comparison workloads.

Run from the repository root, once per thread setting:

    uv run python scripts/clifft_comparison/run_tsim.py --out tsim.jsonl [NAME ...]
    NPROC=1 NUMBA_NUM_THREADS=1 uv run python scripts/clifft_comparison/run_tsim.py \
        --out tsim-1.jsonl

``NPROC=1`` pins JAX/XLA's CPU thread pools to one thread and
``NUMBA_NUM_THREADS=1`` numba's (noise sampling, trajectory sharing). Every
workload is sampled with one lane per shot and with each of its
trajectory-sharing configurations: compile, one untimed warmup call (JIT), then
sample calls for at least ``--seconds``. Counting follows clifft-bench:
attempted shots, accepted shots (no postselected detector fired) and logical
counts (observable 0 among accepted shots). Each configuration appends one JSON line to ``--out``. Every
configuration and thread setting gets its own seed, so pooled counts are
independent.
"""

import argparse
import json
import os
import time
import zlib

import numpy as np
from workloads import WORKLOADS

import tsim
from tsim.statevector.program import StatevectorComponent


def _sample(sampler, workload, shots: int, batch: int, num_detectors: int):
    """Return (accepted shots, observable-0 ones among them) of one call."""
    mask = np.ones(num_detectors, dtype=bool) if workload.postselect else None
    detectors, observables = sampler.sample(
        shots, batch_size=batch, separate_observables=True, postselection_mask=mask
    )
    keep = ~detectors.any(axis=1) if workload.postselect else np.ones(shots, dtype=bool)
    return int(keep.sum()), int(observables[keep, 0].sum())


def measure(name: str, label: str, sharing: bool, batch: int, shots: int, seconds):
    """Compile, warm up and time one configuration; return its JSON record."""
    workload = WORKLOADS[name]
    circuit = tsim.Circuit.from_file(str(workload.path))
    threads = os.environ.get("NPROC", "all")
    seed = zlib.crc32(f"{name}|{label}|{batch}|{shots}|{threads}".encode()) % 2**30
    start = time.perf_counter()
    sampler = circuit.compile_detector_sampler(
        strategy="statevector", seed=seed, trajectory_sharing=sharing
    )
    compile_seconds = time.perf_counter() - start
    components = [
        c for c in sampler._program.components if isinstance(c, StatevectorComponent)
    ]
    start = time.perf_counter()
    _sample(sampler, workload, shots, batch, circuit.num_detectors)
    warmup_seconds = time.perf_counter() - start
    sampler.sharing_stats.clear()
    attempted = accepted = ones = calls = 0
    cpu_start = time.process_time()
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        kept, wrong = _sample(sampler, workload, shots, batch, circuit.num_detectors)
        attempted += shots
        accepted += kept
        ones += wrong
        calls += 1
    elapsed = time.perf_counter() - start
    stats = list(sampler.sharing_stats)
    return {
        "simulator": "tsim",
        "workload": name,
        "configuration": label,
        "threads": threads,
        "seed": seed,
        "batch": batch,
        "shots_per_call": shots,
        "calls": calls,
        "compile_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        "elapsed_seconds": elapsed,
        "shots_per_second": attempted / elapsed,
        "cores_busy": (time.process_time() - cpu_start) / elapsed,
        "attempted": attempted,
        "accepted": accepted,
        "observable_ones": ones,
        "postselect": workload.postselect,
        "width": max((c.width for c in components), default=0),
        "statevector_components": len(components),
        "actions": max((c.num_actions for c in components), default=0),
        "mean_groups": float(np.mean([s["groups"] for s in stats])) if stats else None,
        "mean_peak_lanes": (
            float(np.mean([s["peak lanes"] for s in stats])) if stats else None
        ),
        "sampler": repr(sampler),
    }


def main() -> None:
    """Time the requested workloads and append one JSON line per configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for name in args.names:
        workload = WORKLOADS[name]
        configurations = [("one lane per shot", False, workload.batch, workload.shots)]
        configurations += [
            ("shared", True, batch, shots) for batch, shots in workload.shared
        ]
        for label, sharing, batch, shots in configurations:
            record = measure(name, label, sharing, batch, shots, args.seconds)
            with open(args.out, "a") as out:
                out.write(json.dumps(record) + "\n")
            print(
                f"{name:26s} {label:18s} batch {batch:>7,} "
                f"{record['shots_per_second']:>14,.0f} shots/s "
                f"accept {record['accepted'] / record['attempted']:.4f} "
                f"width {record['width']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
