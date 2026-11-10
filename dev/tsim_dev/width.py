"""`tsim-dev bench --width`: active width of each circuit, compiled but not sampled.

Both sides compile the selected circuits at once, in batches taken by `--jobs`
worker processes (tsim-bench's worker in `width` mode), and each circuit's row is
printed as soon as it has a result on every side. If a worker dies (memory cap) or one circuit compiles for
longer than `--case-timeout`, that circuit is recorded as an error and a new worker
continues with the next one. When the run passes `--timeout`, the circuits not yet
compiled are recorded as errors.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import select
import statistics
import subprocess
import time
from collections.abc import Callable, Iterator
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
    """How a worker ended: its exit code and stderr, or why it was stopped."""

    code: int = 0
    stderr: str = ""
    timed_out: bool = False  # one circuit compiled for longer than --case-timeout
    past_deadline: bool = False  # the whole run passed --timeout


def _stream(
    cmd, env, case_timeout: int, deadline: float, outcome: Exit
) -> Iterator[str]:
    """Run the worker, yield its output lines, and record how it ended in `outcome`.

    The worker is stopped if no line arrives for `case_timeout` seconds (one circuit
    compiles for longer than that), or when the run's `deadline` (monotonic) passes.
    """
    with subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        assert process.stdout is not None and process.stderr is not None
        quiet_until = time.monotonic() + case_timeout
        while True:
            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            now = time.monotonic()
            if ready:
                line = process.stdout.readline()
                if not line:
                    break
                quiet_until = now + case_timeout
                yield line.rstrip("\n")
            elif now > deadline or now > quiet_until:
                process.kill()
                process.wait()
                outcome.past_deadline = now > deadline
                outcome.timed_out = not outcome.past_deadline
                return
        outcome.code = process.wait()
        outcome.stderr = process.stderr.read()


def measure(
    side, cases, args, deadline: float, report: Callable[[str, dict], None]
) -> None:
    """Compile every case on one side and call `report(key, result)` for each one."""
    from .bench import side_env

    pending = _items(cases)
    while pending:
        cmd = [str(side.python), "-u", str(corpus.worker(args.corpus_dir))]
        cmd += ["--mode", "width", "--root", str(corpus.root(args.corpus_dir))]
        if side.strategy or args.strategy:
            cmd += ["--strategy", side.strategy or args.strategy]
        for _, flag, value in pending:
            cmd += [flag, value]
        env = side_env(
            side,
            side.threads or args.threads,
            side.device or getattr(args, "device", "cpu"),
        )
        started, outcome, done = None, Exit(), set()
        capped = common.capped(cmd, mem=args.mem)
        for line in _stream(capped, env, args.case_timeout, deadline, outcome):
            if line.startswith("START "):
                started = line[6:]
            elif line.startswith("RESULT "):
                result = json.loads(line[7:])
                key = result.pop("name")
                done.add(key)
                report(key, result)
        missing = [item for item in pending if item[0] not in done]
        if not missing:
            return
        # NOTE: the worker died or was stopped while compiling `started`.
        if outcome.past_deadline:
            for key, _, _ in missing:
                report(
                    key,
                    {
                        "error": f"not compiled: the run passed --timeout {args.timeout} s"
                    },
                )
            return
        if outcome.timed_out:
            reason = f"compile took over {args.case_timeout} s"
        else:
            reason = common.explain_exit(outcome.code, args.mem, None)
            reason += ": " + (outcome.stderr.strip().splitlines() or [""])[-1]
        error = {"error": reason[:200]}
        if started is None or started not in {key for key, _, _ in missing}:
            for key, _, _ in missing:
                report(key, dict(error))
            return
        report(started, error)
        pending = [item for item in missing if item[0] != started]


BATCH = 8  # circuits per worker process; a free worker takes the next batch


def measure_all(sides, cases, args, deadline: float) -> Iterator[tuple[int, str, dict]]:
    """Compile the cases on every side at once and yield `(side, key, result)` as they come.

    The cases are cut into batches of `BATCH`, small files first, interleaved over the
    sides, and `args.jobs` workers take the next batch whenever they are free. So the
    first rows appear within seconds, one slow compile delays only its own batch, and
    a batch that would start after `deadline` is reported as not compiled at once.
    """
    ordered = sorted(
        cases, key=lambda case: case.path.stat().st_size if case.path.exists() else 0
    )
    batches: queue.Queue = queue.Queue()
    for start in range(0, len(ordered), BATCH):
        for index in range(len(sides)):
            batches.put((index, ordered[start : start + BATCH]))
    results: queue.Queue = queue.Queue()

    def worker() -> None:
        while True:
            try:
                index, batch = batches.get_nowait()
            except queue.Empty:
                return

            def report(key, result, index=index):
                results.put((index, key, result))

            if time.monotonic() > deadline:
                error = f"not compiled: the run passed --timeout {args.timeout} s"
                for key, _, _ in _items(batch):
                    report(key, {"error": error})
                continue
            measure(sides[index], batch, args, deadline, report)

    jobs = max(1, min(args.jobs, batches.qsize()))
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        futures = [pool.submit(worker) for _ in range(jobs)]
        for _ in range(len(cases) * len(sides)):
            while True:
                try:
                    yield results.get(timeout=1.0)
                    break
                except queue.Empty:
                    for future in futures:
                        if future.done():
                            future.result()  # re-raises a worker thread's exception


def _shown(result: dict | None) -> str:
    if result is None or "error" in result:
        return "err"
    width = result.get("width")
    if width is None:
        return "-"
    return f">={width}" if result.get("refused") else str(width)


def _seconds(result: dict) -> str:
    return f"{result['compile_s']:.2f}" if "compile_s" in result else "-"


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
    by_key = {key: case for case, (key, _, _) in zip(cases, _items(cases), strict=True)}
    total = len(by_key)
    header = f"{'':>9} {'circuit':<{NAME}} {'k':>5} {'recorded':>8} {'compile':>8}"
    if has_ref:
        header += f" {'ref k':>6} {'compile':>8}  verdict"
    print(header, flush=True)
    records, counts, partial = [], {}, {}
    # NOTE: rows are printed as circuits finish on every side, not in suite order.
    for index, key, result in measure_all(sides, cases, args, start + args.timeout):
        partial.setdefault(key, {})[index] = result
        if len(partial[key]) < len(sides):
            continue
        case, new = by_key[key], partial[key][0]
        record = {"circuit": case.id or str(case.path), "new": new}
        progress = f"[{len(records) + 1}/{total}]"
        row = f"{progress:>9} {case.name[:NAME]:<{NAME}} {_shown(new):>5}"
        row += f" {'-' if case.tsim_width is None else case.tsim_width!s:>8}"
        row += f" {_seconds(new):>8}"
        if has_ref:
            ref = partial[key][1]
            record["ref"] = ref
            record["verdict"] = verdict(new, ref)
            counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
            row += f" {_shown(ref):>6} {_seconds(ref):>8}"
            row += f"  {record['verdict']}"
        if "error" in new:
            row += f"  ERROR: {new['error']}"
        elif has_ref and "error" in record["ref"]:
            row += f"  ref ERROR: {record['ref']['error']}"
        records.append(record)
        print(row, flush=True)

    elapsed = time.monotonic() - start
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
