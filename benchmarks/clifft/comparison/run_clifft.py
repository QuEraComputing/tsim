"""Time Clifft on the comparison workloads, as a same-machine reference.

Clifft is not a tsim dependency. Install it into a separate environment and run
this script with that interpreter, from the repository root:

    uv venv /tmp/clifft-venv && uv pip install --python /tmp/clifft-venv clifft
    /tmp/clifft-venv/bin/python scripts/clifft_comparison/run_clifft.py --out clifft.jsonl

Every circuit is compiled as clifft-bench does (default HIR passes, every
detector postselected unless the workload says otherwise). For one thread and
for all threads (``threads="auto"``) it samples with ``sample_survivors`` for at
least ``--seconds`` after one untimed call, with scalar execution and with
packed lanes of 2048 shots (unless Clifft rejects that batch as too large).
Each measurement appends one JSON line to ``--out``. Every measurement draws
its own seeds, so pooled counts are independent.
"""

import argparse
import json
import time
import zlib

import clifft
from workloads import WORKLOADS

BATCH_SIZES = (1, 2048)


def measure(
    program, shots: int, seconds: float, threads, batch_size: int, seed: int
) -> dict:
    """Return the timing and counts of one thread setting and batch size."""
    clifft.sample_survivors(
        program, shots, seed=seed, threads=threads, batch_size=batch_size
    )
    attempted = accepted = ones = calls = 0
    cpu_start = time.process_time()
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        calls += 1
        result = clifft.sample_survivors(
            program, shots, seed=seed + calls, threads=threads, batch_size=batch_size
        )
        attempted += int(result.total_shots)
        accepted += int(result.passed_shots)
        ones += int(result.observable_ones[0])
    elapsed = time.perf_counter() - start
    return {
        "calls": calls,
        "elapsed_seconds": elapsed,
        "shots_per_second": attempted / elapsed,
        "cores_busy": (time.process_time() - cpu_start) / elapsed,
        "attempted": attempted,
        "accepted": accepted,
        "observable_ones": ones,
    }


def main() -> None:
    """Time the requested workloads and append one JSON line per measurement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    for name in args.names:
        workload = WORKLOADS[name]
        text = workload.path.read_text()
        start = time.perf_counter()
        hir = clifft.trace(clifft.parse(text))
        clifft.default_hir_pass_manager().run(hir)
        mask = [int(workload.postselect)] * int(hir.num_detectors)
        program = clifft.lower(hir, postselection_mask=mask)
        compile_seconds = time.perf_counter() - start
        shots = workload.clifft_shots or workload.shots
        for threads in (1, "auto"):
            for batch_size in BATCH_SIZES:
                try:
                    key = f"{name}|{threads}|{batch_size}".encode()
                    seed = (zlib.crc32(key) % 2**20) * 2**20
                    record = measure(
                        program, shots, args.seconds, threads, batch_size, seed
                    )
                except ValueError as error:
                    # NOTE: Clifft rejects packed batches above its 64 MiB state limit.
                    print(f"{name} threads={threads} batch={batch_size}: {error}")
                    continue
                record.update(
                    simulator="clifft",
                    version=clifft.__version__,
                    workload=name,
                    threads="1" if threads == 1 else "all",
                    batch=batch_size,
                    shots_per_call=shots,
                    compile_seconds=compile_seconds,
                    postselect=workload.postselect,
                    width=int(program.peak_active_width),
                )
                with open(args.out, "a") as out:
                    out.write(json.dumps(record) + "\n")
                print(
                    f"{name:26s} threads {record['threads']:>3} batch {batch_size:>5} "
                    f"{record['shots_per_second']:>14,.0f} shots/s "
                    f"accept {record['accepted'] / record['attempted']:.4f} "
                    f"k_max {record['width']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
