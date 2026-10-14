"""Benchmark Clifft on the clifft-bench circuits, as a same-machine reference.

Clifft is not a tsim dependency. Install it into a separate environment and run
this script with that interpreter, from the repository root:

    uv venv /tmp/clifft-venv && uv pip install --python /tmp/clifft-venv clifft
    /tmp/clifft-venv/bin/python scripts/statevector/benchmark_clifft.py [--seconds S] [NAME ...]

Every circuit is compiled as clifft-bench does: default HIR passes, every
detector post-selected. Then, for one thread and for all threads
(``threads="auto"``), it samples with ``sample_survivors`` for at least
``--seconds`` after one untimed call, counting attempted shots, accepted shots
and logical errors (observable 0 among accepted shots). Each thread setting is
timed with scalar execution and with packed lanes of 2048 shots (unless Clifft
rejects that batch as too large), and the faster one is reported, as
clifft-bench's batch calibration does. "cores busy" is the process CPU time
divided by wall time over the timed window.
"""

import argparse
import time

import clifft
from workloads import CIRCUITS, WORKLOADS, acceptance_check

BATCH_SIZES = (1, 2048)


def measure(program, shots: int, seconds: float, threads, batch_size: int):
    """Return (shots/s, cores busy, attempted, accepted, logical errors)."""
    clifft.sample_survivors(
        program, shots, seed=0, threads=threads, batch_size=batch_size
    )
    attempted = accepted = errors = calls = 0
    cpu_start = time.process_time()
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        calls += 1
        result = clifft.sample_survivors(
            program, shots, seed=calls, threads=threads, batch_size=batch_size
        )
        attempted += int(result.total_shots)
        accepted += int(result.passed_shots)
        errors += int(result.observable_ones[0])
    elapsed = time.perf_counter() - start
    cores = (time.process_time() - cpu_start) / elapsed
    return attempted / elapsed, cores, attempted, accepted, errors


def main() -> None:
    """Benchmark the requested workloads and print one Markdown row per thread setting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    print(f"Clifft {clifft.__version__}")
    print(
        "| circuit | threads | batch | compile s | k_max | shots/s | cores busy | "
        "acceptance | Clifft 0.10.0 acceptance | check | logical errors / accepted |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for name in args.names:
        file, reference, shots, _ = WORKLOADS[name]
        text = (CIRCUITS / file).read_text()
        start = time.perf_counter()
        hir = clifft.trace(clifft.parse(text))
        clifft.default_hir_pass_manager().run(hir)
        program = clifft.lower(hir, postselection_mask=[1] * int(hir.num_detectors))
        compile_seconds = time.perf_counter() - start
        for threads in (1, "auto"):
            runs = {}
            for batch_size in BATCH_SIZES:
                try:
                    runs[batch_size] = measure(
                        program, shots, args.seconds, threads, batch_size
                    )
                except ValueError:
                    # NOTE: Clifft rejects packed batches above its 64 MiB state limit.
                    continue
            batch_size, (rate, cores, attempted, accepted, errors) = max(
                runs.items(), key=lambda item: item[1][0]
            )
            acceptance = accepted / attempted
            check = acceptance_check(acceptance, reference, attempted)
            print(
                f"| {name} | {threads} | {batch_size} | {compile_seconds:.2f} | "
                f"{program.peak_active_width} | {rate:,.0f} | {cores:.1f} | "
                f"{acceptance:.4f} | {reference:.4f} | {check} | "
                f"{errors} / {accepted} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
