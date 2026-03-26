"""Benchmark tsim vs stim detector sampling on .stim circuit files.

Runs batch-size autotuning (doubling loop) for both tsim and stim on every
circuit listed in CIRCUITS, then writes an enriched JSON with per-circuit
timing data and full hardware metadata.

Usage::

    uv run python benchmarks/stim/run_vs_stim.py --machine "M4"
    uv run python benchmarks/stim/run_vs_stim.py --machine "GH200" --output benchmarks/stim/results_gpu.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import stim
from pyzx_param.simulate import DecompositionStrategy

import tsim

try:
    from benchmarks.metadata import collect_metadata
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metadata import collect_metadata


CIRCUITS = [
    # {
    #     "file": "benchmarks/stim/d=5_X.stim",
    #     "label": "d=5 distillation",
    #     "publication": "Rodriguez et al. 2025",
    #     "strategy": "cat5",
    # },
    # {
    #     "file": "benchmarks/stim/d=3_X.stim",
    #     "label": "d=3 distillation",
    #     "publication": "Rodriguez et al. 2025",
    #     "strategy": "cat5",
    # },
    # {
    #     "file": "benchmarks/stim/cultivation_d=3.stim",
    #     "label": "d=3 cultivation",
    #     "publication": "arXiv:2409.17595",
    #     "strategy": "cutting",
    # },
    {
        "file": "benchmarks/stim/d=3-degenerate-basis=Y-p=0.001.stim",
        "label": "d=3 cultivation",
        "publication": "arXiv:2409.17595",
        "strategy": "cutting",
    },
    {
        "file": "benchmarks/stim/d=3-degenerate-basis=Y-p=0.001_T.stim",
        "label": "d=3 cultivation",
        "publication": "arXiv:2409.17595",
        "strategy": "cutting",
    },
]


def _save_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Autotuning workers (run in subprocesses for crash safety)
# ---------------------------------------------------------------------------


def _tsim_worker(
    progress_path: str,
    stim_file: str,
    initial_batch_size: int,
    convergence_ratio: float,
    strategy: DecompositionStrategy = "cat5",
) -> None:
    circuit = tsim.Circuit.from_file(stim_file)
    sampler = circuit.compile_detector_sampler(strategy=strategy)
    print(f"  [tsim] strategy={strategy}  {sampler}")
    _autotune_detector_sampler(
        progress_path,
        sampler,
        initial_batch_size,
        convergence_ratio,
    )


def _stim_worker(
    progress_path: str,
    stim_file: str,
    initial_batch_size: int,
    convergence_ratio: float,
) -> None:
    circuit = tsim.Circuit.from_file(stim_file).stim_circuit
    sampler = circuit.compile_detector_sampler()
    _autotune_detector_sampler(
        progress_path,
        sampler,
        initial_batch_size,
        1.0,
    )


def _autotune_detector_sampler(
    progress_path: str,
    sampler: Any,
    initial_batch_size: int,
    convergence_ratio: float,
) -> None:
    """Doubling-loop autotuner identical to benchmarks/run.py."""
    best_duration = float("inf")
    best_batch_size = initial_batch_size
    batch_size = initial_batch_size
    probes: list[dict[str, float | int]] = []
    ppath = Path(progress_path)

    while True:
        sampler.sample(batch_size, append_observables=True)

        repeats = 0
        start = time.perf_counter()
        while True:
            repeats += 1
            sampler.sample(batch_size, append_observables=True)
            if time.perf_counter() - start > 1:
                break
        duration = (time.perf_counter() - start) / batch_size / repeats

        probes.append({"batch_size": batch_size, "duration_per_shot": duration})
        print(f"    batch_size={batch_size:>8,}  " f"duration={duration:.3e} s/shot")

        converged = duration / best_duration > convergence_ratio
        if duration < best_duration:
            best_duration = duration
            best_batch_size = batch_size

        ppath.write_text(
            json.dumps(
                {
                    "best_duration": best_duration,
                    "best_batch_size": best_batch_size,
                    "probes": probes,
                }
            ),
            encoding="utf-8",
        )

        if converged:
            print(f"    converged (ratio {duration / best_duration:.2f})")
            break
        batch_size *= 2


def _run_in_subprocess(
    target: Any,
    stim_file: str,
    initial_batch_size: int,
    convergence_ratio: float,
    **extra_kwargs: Any,
) -> dict[str, Any] | None:
    fd, progress_path = tempfile.mkstemp(prefix="tsim_stim_probe_", suffix=".json")
    os.close(fd)
    try:
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=target,
            args=(progress_path, stim_file, initial_batch_size, convergence_ratio),
            kwargs=extra_kwargs,
        )
        proc.start()
        proc.join()

        ppath = Path(progress_path)
        if ppath.stat().st_size == 0:
            if proc.exitcode != 0:
                print(f"    [child crashed (exit {proc.exitcode}) before any result]")
            return None

        result = json.loads(ppath.read_text(encoding="utf-8"))
        if proc.exitcode != 0:
            print(
                f"    [child crashed (exit {proc.exitcode}), "
                f"using last good batch_size={result['best_batch_size']}]"
            )
        return result
    finally:
        Path(progress_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------


def run_benchmarks(
    *,
    machine_label: str,
    output_path: Path,
    initial_batch_size: int = 32,
    convergence_ratio: float = 1.2,
    circuits: list[dict[str, str]] | None = None,
) -> Path:
    """Run tsim and stim benchmarks for all circuits and save results."""
    if circuits is None:
        circuits = CIRCUITS

    metadata = collect_metadata(machine_label)
    run_metadata: dict[str, Any] = {
        "benchmark_name": "tsim_vs_stim_detector_sampling",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "machine": metadata,
        "parameters": {
            "initial_batch_size": initial_batch_size,
            "convergence_ratio": convergence_ratio,
        },
    }

    records: list[dict[str, Any]] = []

    for idx, entry in enumerate(circuits):
        stim_file = entry["file"]
        label = entry["label"]
        strategy = entry.get("strategy", "cat5")
        print(
            f"\n[{idx + 1}/{len(circuits)}] {label}  ({stim_file})  strategy={strategy}"
        )

        record: dict[str, Any] = {
            "file": stim_file,
            "label": label,
            "publication": entry.get("publication", ""),
            "strategy": strategy,
        }

        # --- tsim ---
        print("  tsim:")
        tsim_result = _run_in_subprocess(
            _tsim_worker,
            stim_file,
            initial_batch_size,
            convergence_ratio,
            strategy=strategy,
        )
        if tsim_result is not None:
            record["tsim"] = {
                "status": "ok",
                "best_duration_per_shot_secs": tsim_result["best_duration"],
                "best_batch_size": tsim_result["best_batch_size"],
                "probes": tsim_result["probes"],
            }
        else:
            record["tsim"] = {"status": "crashed"}

        # --- stim ---
        print("  stim:")
        stim_result = _run_in_subprocess(
            _stim_worker, stim_file, initial_batch_size, convergence_ratio
        )
        if stim_result is not None:
            record["stim"] = {
                "status": "ok",
                "best_duration_per_shot_secs": stim_result["best_duration"],
                "best_batch_size": stim_result["best_batch_size"],
                "probes": stim_result["probes"],
            }
        else:
            record["stim"] = {"status": "crashed"}

        records.append(record)

        incremental = {
            "run_metadata": {
                **run_metadata,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "records_total": len(circuits),
                "records_done": len(records),
            },
            "records": records,
        }
        _save_json_atomic(output_path, incremental)
        print(f"  [saved {len(records)}/{len(circuits)} to {output_path}]")

    final = {
        "run_metadata": {
            **run_metadata,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "records_total": len(circuits),
            "records_done": len(records),
        },
        "records": records,
    }
    _save_json_atomic(output_path, final)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark tsim vs stim detector sampling on .stim circuits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--machine",
        required=True,
        help="Human-readable machine label (e.g. 'M4 Pro MacBook').",
    )
    p.add_argument(
        "--output",
        default="",
        help="Output JSON path. Default: benchmarks/stim/results_<machine>.json",
    )
    p.add_argument("--initial-batch-size", type=int, default=32)
    p.add_argument("--convergence-ratio", type=float, default=1.5)
    return p


def main(argv: list[str] | None = None) -> Path:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)

    if args.output:
        output_path = Path(args.output)
    else:
        safe_machine = args.machine.replace(" ", "_").lower()
        output_path = Path(f"benchmarks/stim/results_{safe_machine}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Machine: {args.machine}")
    print(f"Output:  {output_path}")

    result = run_benchmarks(
        machine_label=args.machine,
        output_path=output_path,
        initial_batch_size=args.initial_batch_size,
        convergence_ratio=args.convergence_ratio,
    )
    print(f"\nResults saved to {result}")
    return result


if __name__ == "__main__":
    main()
