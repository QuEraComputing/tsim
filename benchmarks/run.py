"""Run benchmarks and save results to JSON.

Usage::

    uv run python -m benchmarks.run --machine "M4 Pro MacBook" --benchmark t_count_scaling
    uv run python -m benchmarks.run --machine "A100 GPU" --t-count-max 40 --repetitions 5

Results are written to ``benchmarks/results/`` as JSON files that include
full machine metadata, making it easy to compare runs across hardware.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
import time
import typing
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.circuits import random_clifford_t
from benchmarks.metadata import collect_metadata

# ---------------------------------------------------------------------------
# Timing helper (subprocess-isolated)
# ---------------------------------------------------------------------------


def _find_throughput_worker(
    progress_path: str,
    n_qubits: int,
    t_count: int,
    seed: int,
    initial_batch_size: int,
    convergence_ratio: float,
) -> None:
    """Child-process target: compile sampler, run doubling loop, write results.

    After each successful batch-size probe the current best
    ``(duration_per_shot, batch_size)`` is written to *progress_path* so the
    parent can recover the last good result if this process crashes
    (e.g. LLVM OOM).
    """
    circuit = random_clifford_t(n_qubits=n_qubits, t_count=t_count, seed=seed)
    sampler = circuit.compile_sampler()

    best_duration = float("inf")
    batch_size = initial_batch_size
    ppath = Path(progress_path)
    probes: list[dict] = []

    while True:
        # Warmup (ensures JIT compilation / caching)
        sampler.sample(batch_size, batch_size=batch_size)

        # Timed run
        start = time.perf_counter()
        sampler.sample(batch_size, batch_size=batch_size)
        duration = (time.perf_counter() - start) / batch_size

        probes.append({"batch_size": batch_size, "duration_per_shot": duration})

        converged = duration / best_duration > convergence_ratio
        best_duration = min(best_duration, duration)

        # Flush progress so the parent can recover after a crash.
        ppath.write_text(
            json.dumps(
                {
                    "best_duration": best_duration,
                    "best_batch_size": batch_size,
                    "probes": probes,
                }
            )
        )

        if converged:
            break

        batch_size *= 2


def find_throughput(
    n_qubits: int,
    t_count: int,
    seed: int,
    *,
    initial_batch_size: int = 32,
    convergence_ratio: float = 0.8,
) -> dict | None:
    """Find optimal batch size and measure per-shot duration.

    The entire doubling loop runs in a **child process** so that a fatal
    LLVM/XLA OOM crash does not kill the parent.  After each successful
    probe the child writes its progress to a temporary file; the parent
    reads the last written value and deletes the file.

    Returns:
        A dict with keys ``best_duration``, ``best_batch_size``, and
        ``probes`` (a list of ``{"batch_size": int, "duration_per_shot":
        float}`` for every doubling step), or ``None`` if the child crashed
        before producing any result.

    """
    # Create a temp file for the child to write progress into.
    fd, progress_path = tempfile.mkstemp(prefix="tsim_probe_", suffix=".json")
    os.close(fd)

    try:
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_find_throughput_worker,
            args=(
                progress_path,
                n_qubits,
                t_count,
                seed,
                initial_batch_size,
                convergence_ratio,
            ),
        )
        proc.start()
        proc.join()

        # Read whatever the child managed to write before exiting / crashing.
        ppath = Path(progress_path)
        if ppath.stat().st_size == 0:
            # Child crashed before writing anything.
            if proc.exitcode != 0:
                print(
                    f"    [child crashed (exit {proc.exitcode}) "
                    f"before producing any result — skipping]"
                )
            return None

        result = json.loads(ppath.read_text())

        if proc.exitcode != 0:
            print(
                f"    [child crashed (exit {proc.exitcode}), "
                f"using last good batch_size={result['best_batch_size']}]"
            )

        return result
    finally:
        # Always clean up the temp file.
        Path(progress_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Benchmark: T-count scaling
# ---------------------------------------------------------------------------


def t_count_scaling(
    *,
    n_qubits: int = 20,
    t_count_min: int = 1,
    t_count_max: int = 20,
    repetitions: int = 100,
    initial_batch_size: int = 32,
    on_t_count_done: typing.Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    """Benchmark sampling throughput as a function of T-count.

    For each T-count in ``[t_count_min, t_count_max]`` and each repetition
    (different random seed), compiles a sampler and finds the optimal batch
    size / per-shot duration.

    If *on_t_count_done* is provided it is called with the accumulated results
    list after all repetitions of each T-count value complete, allowing the
    caller to save incremental progress.
    """
    results: list[dict] = []

    for t_count in range(t_count_min, t_count_max + 1):
        print(f"T-count: {t_count}")
        for rep in range(repetitions):
            result = find_throughput(
                n_qubits,
                t_count,
                rep,
                initial_batch_size=initial_batch_size,
            )
            if result is None:
                continue
            results.append(
                {
                    "t_count": t_count,
                    "seed": rep,
                    "duration_per_shot": result["best_duration"],
                    "best_batch_size": result["best_batch_size"],
                    "probes": result["probes"],
                }
            )
            print(
                f"  rep={rep}  batch_size={result['best_batch_size']:>8,}  "
                f"duration={result['best_duration']:.3e} s/shot"
            )

        if on_t_count_done is not None:
            on_t_count_done(results)

    return results


# ---------------------------------------------------------------------------
# Registry of available benchmarks
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, typing.Callable[..., list[dict]]] = {
    "t_count_scaling": t_count_scaling,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run tsim benchmarks and save results to JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--machine",
        required=True,
        help="Human-readable machine label (e.g. 'M4 Pro MacBook')",
    )
    p.add_argument(
        "--benchmark",
        default="t_count_scaling",
        choices=list(BENCHMARKS),
        help="Which benchmark to run",
    )
    p.add_argument("--n-qubits", type=int, default=20)
    p.add_argument("--t-count-min", type=int, default=1)
    p.add_argument("--t-count-max", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=10)
    p.add_argument("--initial-batch-size", type=int, default=100)
    p.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to write result JSON files",
    )
    return p


def _save_payload(filepath: Path, payload: dict) -> None:
    """Atomically overwrite *filepath* with the JSON-encoded *payload*."""
    tmp = filepath.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(filepath)


def main(argv: list[str] | None = None) -> Path:
    """Run benchmark and save results. Returns the output file path."""
    args = _build_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output path before starting so we can save incrementally.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_machine = args.machine.replace(" ", "_").lower()
    filename = f"{args.benchmark}_{safe_machine}_{timestamp}.json"
    filepath = output_dir / filename

    metadata = collect_metadata(args.machine)
    payload: dict = {
        "metadata": metadata,
        "benchmark": args.benchmark,
        "parameters": {
            "n_qubits": args.n_qubits,
            "t_count_min": args.t_count_min,
            "t_count_max": args.t_count_max,
            "repetitions": args.repetitions,
            "initial_batch_size": args.initial_batch_size,
        },
        "results": [],
    }

    def _on_progress(results: list[dict]) -> None:
        """Flush current results to disk after each T-count."""
        payload["results"] = results
        _save_payload(filepath, payload)
        print(f"  [saved {len(results)} data points to {filepath}]")

    print(f"Running benchmark: {args.benchmark}")
    print(f"Machine: {args.machine}")
    print(f"Output:  {filepath}")
    print()

    bench_fn = BENCHMARKS[args.benchmark]
    results = bench_fn(
        n_qubits=args.n_qubits,
        t_count_min=args.t_count_min,
        t_count_max=args.t_count_max,
        repetitions=args.repetitions,
        initial_batch_size=args.initial_batch_size,
        on_t_count_done=_on_progress,
    )

    # Final save (in case the callback list reference diverged).
    payload["results"] = results
    _save_payload(filepath, payload)

    print(f"\nResults saved to {filepath}")
    return filepath


if __name__ == "__main__":
    main()
