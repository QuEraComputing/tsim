"""`tsim-dev bench --width`: active width of each circuit, compiled but not sampled.

Each side compiles all selected circuits in one worker process (tsim-bench's worker in
`width` mode), so the whole corpus takes minutes rather than hours. If the process dies
(memory cap) or one circuit compiles for longer than `--case-timeout`, that circuit is
recorded as an error and a new process continues with the next one.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import select
import statistics
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass

from . import common, corpus
from .common import FAILED, INFRA, OK

NAME = 44


def _items(cases) -> list[tuple[str, str, str]]:
    """Return (key, flag, value) for each case: `--case ID` or `--circuit PATH`."""
    return [
        (
            (case.id, "--case", case.id)
            if case.id
            else (str(case.path), "--circuit", str(case.path))
        )
        for case in cases
    ]


@dataclass
class Exit:
    """How a worker ended: its exit code and stderr, or a stop for taking too long."""

    code: int = 0
    stderr: str = ""
    timed_out: bool = False


def _stream(cmd, env, case_timeout: int, outcome: Exit) -> Iterator[str]:
    """Run the worker, yield its output lines, and record how it ended in `outcome`.

    The worker is stopped if no line arrives for `case_timeout` seconds, that is if
    one circuit compiles for longer than that.
    """
    with subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        assert process.stdout is not None and process.stderr is not None
        deadline = time.monotonic() + case_timeout
        while True:
            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            if ready:
                line = process.stdout.readline()
                if not line:
                    break
                deadline = time.monotonic() + case_timeout
                yield line.rstrip("\n")
            elif time.monotonic() > deadline:
                process.kill()
                process.wait()
                outcome.timed_out = True
                return
        outcome.code = process.wait()
        outcome.stderr = process.stderr.read()


def measure(side, cases, args) -> dict[str, dict]:
    """Compile every case on one side and return its result by case key."""
    from .bench import side_env

    results: dict[str, dict] = {}
    pending = _items(cases)
    while pending:
        cmd = [str(side.python), "-u", str(corpus.worker(args.corpus_dir))]
        cmd += ["--mode", "width", "--root", str(corpus.root(args.corpus_dir))]
        if side.strategy or args.strategy:
            cmd += ["--strategy", side.strategy or args.strategy]
        for _, flag, value in pending:
            cmd += [flag, value]
        env = side_env(side, side.threads or args.threads)
        started, outcome = None, Exit()
        capped = common.capped(cmd, mem=args.mem)
        for line in _stream(capped, env, args.case_timeout, outcome):
            if line.startswith("START "):
                started = line[6:]
            elif line.startswith("RESULT "):
                result = json.loads(line[7:])
                results[result.pop("name")] = result
        missing = [item for item in pending if item[0] not in results]
        if not missing:
            break
        # NOTE: the worker died or was stopped while compiling `started`.
        if outcome.timed_out:
            reason = f"compile took over {args.case_timeout} s"
        else:
            reason = common.explain_exit(outcome.code, args.mem, None)
            reason += ": " + (outcome.stderr.strip().splitlines() or [""])[-1]
        error = {"error": reason[:200]}
        if started is None or started not in {key for key, _, _ in missing}:
            results.update({key: dict(error) for key, _, _ in missing})
            break
        results[started] = error
        pending = [item for item in missing if item[0] != started]
    return results


def measure_parallel(side, cases, args) -> dict[str, dict]:
    """Split the cases over `args.jobs` workers, largest files first, and merge."""
    ordered = sorted(cases, key=lambda case: -case.path.stat().st_size)
    jobs = max(1, min(args.jobs, len(ordered)))
    chunks = [ordered[i::jobs] for i in range(jobs)]
    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        for part in pool.map(lambda chunk: measure(side, chunk, args), chunks):
            results.update(part)
    return results


def _shown(result: dict | None) -> str:
    if result is None or "error" in result:
        return "err"
    width = result.get("width")
    if width is None:
        return "-"
    return f">={width}" if result.get("refused") else str(width)


def verdict(new: dict, ref: dict) -> str:
    """Compare two widths, lower is better; a refused width is only a lower bound."""
    if "error" in new or "error" in ref:
        return "error"
    a, b = new.get("width"), ref.get("width")
    if a is None or b is None:
        return "same" if a == b else "n/a"
    if new.get("refused") and ref.get("refused"):
        return "same" if a == b else "unknown"  # first refused component, not the max
    if new.get("refused"):
        return "wider" if a > b else "unknown"
    if ref.get("refused"):
        return "narrower" if a < b else "unknown"
    return "same" if a == b else ("narrower" if a < b else "wider")


def run(args, cases, sides) -> int:
    """Compile every case on each side, print the widths and the verdicts."""
    has_ref = len(sides) == 2
    print(f"bench --width: new={sides[0].describe()}", end="")
    print(f" ref={sides[1].describe()}" if has_ref else "", end="")
    print(
        f" circuits={len(cases)} tsim-bench={corpus.describe(args.corpus_dir)}",
        flush=True,
    )
    start = time.monotonic()
    per_side = [measure_parallel(side, cases, args) for side in sides]
    elapsed = time.monotonic() - start
    keys = [key for key, _, _ in _items(cases)]

    header = f"{'circuit':<{NAME}} {'k':>5} {'recorded':>8} {'compile':>8}"
    if has_ref:
        header += f" {'ref k':>6} {'compile':>8}  verdict"
    print(header)
    records, counts = [], {}
    for case, key in zip(cases, keys, strict=True):
        new = per_side[0][key]
        record = {"circuit": case.id or str(case.path), "new": new}
        row = f"{case.name[:NAME]:<{NAME}} {_shown(new):>5}"
        row += f" {'-' if case.tsim_width is None else case.tsim_width!s:>8}"
        row += f" {new.get('compile_s', float('nan')):>8.2f}"
        if has_ref:
            ref = per_side[1][key]
            record["ref"] = ref
            record["verdict"] = verdict(new, ref)
            counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
            row += f" {_shown(ref):>6} {ref.get('compile_s', float('nan')):>8.2f}"
            row += f"  {record['verdict']}"
        if "error" in new:
            row += f"  ERROR: {new['error']}"
        elif has_ref and "error" in record["ref"]:
            row += f"  ref ERROR: {record['ref']['error']}"
        records.append(record)
        print(row, flush=True)

    out = common.out_dir("bench") / f"width-{common.stamp()}.json"
    meta = {
        "new": sides[0].describe(),
        "ref": sides[1].describe() if has_ref else None,
        "tsim_bench": corpus.describe(args.corpus_dir),
        "cpu_count": os.cpu_count(),
    }
    out.write_text(json.dumps({"meta": meta, "results": records}, indent=1))

    errors = sum("error" in r["new"] or "error" in r.get("ref", {}) for r in records)
    compiles = [r["new"]["compile_s"] for r in records if "compile_s" in r["new"]]
    summary = f"SUMMARY: {len(records)} circuits, {errors} errors in {elapsed:.0f} s"
    if compiles:
        summary += f", median compile {statistics.median(compiles):.2f} s"
    if has_ref:
        summary += ", " + ", ".join(
            f"{counts.get(v, 0)} {v}" for v in ("wider", "narrower", "same", "unknown")
        )
    print(f"{summary}. Details: {out}")
    if errors:
        return INFRA
    return FAILED if counts.get("wider") else OK
