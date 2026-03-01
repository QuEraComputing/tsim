"""Benchmark throughput for QASM circuits stored in a JSON file.

Workflow:
1. Load records from a JSON file (list[dict] or dict).
2. Convert each record's QASM circuit to TSIM/STIM program text.
3. Optionally append terminal measurements on all qubits.
4. Benchmark throughput with the same doubling strategy as benchmarks/run.py.
5. Write an enriched JSON that preserves original record fields and adds
   per-circuit TSIM benchmarking metadata.
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

import tsim

try:
    # Module execution: uv run python -m benchmarks.quizx_tsim_bench ...
    from benchmarks.metadata import collect_metadata
    from benchmarks.qasm_to_tsim import qasm_to_tsim_program
except ModuleNotFoundError:
    # Script-path execution: uv run python benchmarks/quizx_tsim_bench.py ...
    from metadata import collect_metadata
    from qasm_to_tsim import qasm_to_tsim_program


def _save_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write JSON payload to disk."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _append_measurements(program: str, n_qubits: int) -> str:
    """Append a terminal measurement over all qubits."""
    if n_qubits <= 0:
        return program
    idx = " ".join(str(i) for i in range(n_qubits))
    lines = [program.rstrip(), f"M {idx}"]
    return "\n".join(lines).strip() + "\n"


def _find_throughput_worker(
    progress_path: str,
    stim_program_text: str,
    sample_seed: int | None,
    initial_batch_size: int,
    convergence_ratio: float,
) -> None:
    """Run throughput search in a child process and flush progress to disk."""
    circuit = tsim.Circuit(stim_program_text)
    sampler = circuit.compile_sampler(seed=sample_seed)
    print(
        f"Number of measurements: {circuit.num_measurements}, num t gates: {circuit.tcount()}"
    )
    print(sampler)

    best_duration = float("inf")
    best_batch_size = initial_batch_size
    batch_size = initial_batch_size
    probes: list[dict[str, float | int]] = []
    ppath = Path(progress_path)

    while True:
        # Warmup (JIT + caches)
        sampler.sample(batch_size, batch_size=batch_size)

        # Timed run
        repeats = 0
        start = time.perf_counter()
        while True:
            repeats += 1
            sampler.sample(batch_size, batch_size=batch_size)
            if time.perf_counter() - start > 1:
                break
        duration = (time.perf_counter() - start) / batch_size / repeats

        probes.append({"batch_size": batch_size, "duration_per_shot": duration})
        print(f"Batch size: {batch_size}, Duration per shot: {duration * 1e6:.2f} us")

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
            print(f"Converged with ratio {duration / best_duration:.2f}")
            break
        batch_size *= 2


def find_throughput_for_program(
    stim_program_text: str,
    *,
    sample_seed: int | None = None,
    initial_batch_size: int = 32,
    convergence_ratio: float = 0.8,
) -> dict[str, Any] | None:
    """Measure throughput for one TSIM/STIM program using crash-safe subprocess."""
    fd, progress_path = tempfile.mkstemp(prefix="tsim_program_probe_", suffix=".json")
    os.close(fd)
    try:
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_find_throughput_worker,
            args=(
                progress_path,
                stim_program_text,
                sample_seed,
                initial_batch_size,
                convergence_ratio,
            ),
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


def _coerce_records(payload: Any) -> tuple[list[dict[str, Any]], str]:
    """Normalize input payload into records and remember source shape."""
    if isinstance(payload, list):
        return payload, "list"
    if isinstance(payload, dict):
        return [payload], "dict"
    raise TypeError("Input JSON must be either a list or an object")


def _resolve_output_path(input_path: Path, output_path: str | None) -> Path:
    """Return output path, defaulting to <input>_tsim_bench.json."""
    if output_path:
        return Path(output_path)
    return input_path.with_name(f"{input_path.stem}_tsim_bench{input_path.suffix}")


def benchmark_quizx_json(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    machine_label: str = "unspecified",
    qasm_key: str = "circuit_qasm",
    include_tsim_program: bool = True,
    append_measurements: bool = True,
    sample_seed_key: str = "seed",
    initial_batch_size: int = 32,
    convergence_ratio: float = 0.8,
    max_circuits: int | None = None,
) -> Path:
    """Run QASM->TSIM conversion + throughput tuning for each circuit record."""
    input_path = Path(input_path)
    output_path = _resolve_output_path(
        input_path, None if output_path is None else str(output_path)
    )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records, shape = _coerce_records(payload)
    if max_circuits is not None:
        records = records[:max_circuits]

    env_metadata = collect_metadata(machine_label)
    run_metadata: dict[str, Any] = {
        "benchmark_name": "quizx_qasm_to_tsim_throughput",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "machine": env_metadata,
        "parameters": {
            "qasm_key": qasm_key,
            "include_tsim_program": include_tsim_program,
            "append_measurements": append_measurements,
            "sample_seed_key": sample_seed_key,
            "initial_batch_size": initial_batch_size,
            "convergence_ratio": convergence_ratio,
            "max_circuits": max_circuits,
        },
    }

    enriched_records: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        print(f"[{idx + 1}/{len(records)}] benchmarking seed={record.get('seed')}")
        out_record = dict(record)

        try:
            qasm = str(record[qasm_key])
        except KeyError as exc:
            out_record["tsim_benchmark"] = {
                "status": "error",
                "error": f"Missing key {qasm_key!r}: {exc}",
            }
            enriched_records.append(out_record)
            continue

        try:
            tsim_program = qasm_to_tsim_program(qasm)
            n_qubits = int(record.get("qubits", tsim.Circuit(tsim_program).num_qubits))
            bench_program = (
                _append_measurements(tsim_program, n_qubits)
                if append_measurements
                else tsim_program
            )
            sample_seed_raw = record.get(sample_seed_key)
            sample_seed = int(sample_seed_raw) if sample_seed_raw is not None else None

            result = find_throughput_for_program(
                bench_program,
                sample_seed=sample_seed,
                initial_batch_size=initial_batch_size,
                convergence_ratio=convergence_ratio,
            )
            if include_tsim_program:
                out_record["tsim_program"] = tsim_program

            if result is None:
                out_record["tsim_benchmark"] = {
                    "status": "crashed",
                    "sample_seed": sample_seed,
                    "initial_batch_size": initial_batch_size,
                    "convergence_ratio": convergence_ratio,
                    "measurement_appended": append_measurements,
                    "best_duration_per_shot_secs": None,
                    "best_batch_size": None,
                    "probes": [],
                }
            else:
                out_record["tsim_benchmark"] = {
                    "status": "ok",
                    "sample_seed": sample_seed,
                    "initial_batch_size": initial_batch_size,
                    "convergence_ratio": convergence_ratio,
                    "measurement_appended": append_measurements,
                    "best_duration_per_shot_secs": result["best_duration"],
                    "best_batch_size": result["best_batch_size"],
                    "probes": result["probes"],
                }
        except Exception as exc:
            out_record["tsim_benchmark"] = {
                "status": "error",
                "error": str(exc),
            }

        enriched_records.append(out_record)

        # Incremental save after each circuit.
        incremental = {
            "run_metadata": {
                **run_metadata,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "records_total": len(records),
                "records_done": len(enriched_records),
                "source_shape": shape,
            },
            "records": enriched_records,
        }
        _save_json_atomic(output_path, incremental)

    final_payload = {
        "run_metadata": {
            **run_metadata,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "records_total": len(records),
            "records_done": len(enriched_records),
            "source_shape": shape,
        },
        "records": enriched_records,
    }
    _save_json_atomic(output_path, final_payload)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert QASM circuits in JSON to TSIM/STIM and benchmark throughput."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to input JSON file.")
    p.add_argument(
        "--output",
        default="",
        help="Optional output path (default: <input>_tsim_bench.json).",
    )
    p.add_argument(
        "--machine",
        default="unspecified",
        help="Human-readable machine label for benchmark metadata.",
    )
    p.add_argument("--qasm-key", default="circuit_qasm")
    p.add_argument("--sample-seed-key", default="seed")
    p.add_argument("--initial-batch-size", type=int, default=32)
    p.add_argument("--convergence-ratio", type=float, default=1)
    p.add_argument(
        "--max-circuits",
        type=int,
        default=-1,
        help="Benchmark only the first N circuits; -1 means all.",
    )
    p.add_argument(
        "--include-tsim-program",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include converted TSIM program text in each output record.",
    )
    p.add_argument(
        "--append-measurements",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append terminal measurement over all qubits before benchmarking.",
    )
    return p


def main(argv: list[str] | None = None) -> Path:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)
    output = benchmark_quizx_json(
        input_path=args.input,
        output_path=args.output if args.output else None,
        machine_label=args.machine,
        qasm_key=args.qasm_key,
        include_tsim_program=args.include_tsim_program,
        append_measurements=args.append_measurements,
        sample_seed_key=args.sample_seed_key,
        initial_batch_size=args.initial_batch_size,
        convergence_ratio=args.convergence_ratio,
        max_circuits=None if args.max_circuits < 0 else args.max_circuits,
    )
    print(f"\nSaved enriched benchmark JSON to {output}")
    return output


if __name__ == "__main__":
    main()
