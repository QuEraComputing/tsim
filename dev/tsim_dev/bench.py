"""`tsim-dev bench`: throughput and compile time, optionally against a reference.

Every measurement runs in a fresh process under a memory cap, pinned to fixed
cores, with numba's on-disk cache warm. With `--ref`, the two checkouts are
measured in interleaved rounds and compared round by round.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import common
from .common import FAILED, INFRA, OK, InfraError


@dataclass(frozen=True)
class Case:
    """One circuit with the sampler options it is measured with."""

    name: str
    path: Path
    sampler: str = "auto"
    postselect: str = "none"


@dataclass(frozen=True)
class Side:
    """One checkout under test: a label and the Python that imports its Tsim."""

    label: str
    python: Path


def registry() -> dict[str, tuple[Case, str]]:
    """Return the named circuits in `dev/circuits/circuits.toml` with their suite."""
    with open(common.CIRCUITS_DIR / "circuits.toml", "rb") as file:
        table = tomllib.load(file)
    return {
        name: (
            Case(
                name,
                common.CIRCUITS_DIR / entry["file"],
                entry.get("sampler", "auto"),
                entry.get("postselect", "none"),
            ),
            entry.get("suite", "full"),
        )
        for name, entry in table.items()
    }


def corpus_dir(explicit: str | None = None) -> Path:
    """Return the tsim-bench checkout to read circuits from."""
    for candidate in (
        explicit,
        os.environ.get("TSIM_BENCH_DIR"),
        common.cache_dir() / "tsim-bench",
    ):
        if candidate and (Path(candidate).expanduser() / "papers").is_dir():
            return Path(candidate).expanduser()
    raise InfraError(
        "no tsim-bench corpus: run `tsim-dev corpus fetch`, or set TSIM_BENCH_DIR "
        "to a clone of https://github.com/rafaelha/tsim-bench"
    )


def corpus_cases(directory: Path, max_k: int) -> list[Case]:
    """Return the corpus circuits whose recorded active width is at most `max_k`."""
    cases = []
    for record in sorted(directory.glob("papers/*/k_max.json")):
        for entry in json.loads(record.read_text()):
            k = entry.get("tsim_k")
            path = record.parent / entry["file"]
            if isinstance(k, int) and k <= max_k and path.exists():
                cases.append(Case(f"{record.parent.name[:10]}/{path.stem}", path))
    return cases


def select(args) -> list[Case]:
    """Turn the command line into the list of cases to measure."""
    named = registry()
    if args.corpus is not None:
        cases = corpus_cases(corpus_dir(args.corpus or None), args.max_k)
    elif args.circuits:
        cases = []
        for item in args.circuits:
            if item in named:
                cases.append(named[item][0])
            elif Path(item).is_file():
                path = Path(item).resolve()
                cases.append(Case(path.stem, path, args.sampler, args.postselect))
            else:
                raise InfraError(
                    f"unknown circuit {item!r}: give a .stim path or one of "
                    + ", ".join(named)
                )
    else:
        cases = [
            case
            for case, suite in named.values()
            if args.suite == "full" or suite == "quick"
        ]
    if args.only:
        cases = [case for case in cases if args.only in case.name]
    if not cases:
        raise InfraError("no circuits selected")
    return cases


def reference(ref: str) -> Side:
    """Return the side for a git revision or for the path of another checkout.

    A revision is exported once into the shared cache and gets its own
    environment there, so its numba cache is separate from the working tree's.
    """
    if Path(ref).is_dir():
        path = Path(ref).resolve()
        python = path / ".venv" / "bin" / "python"
        if not python.exists():
            raise InfraError(f"{path} has no .venv; run `uv sync` there")
        return Side(common.describe(path), python)
    sha = common.git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=common.root())
    target = common.cache_dir() / "refs" / sha[:12]
    python = target / ".venv" / "bin" / "python"
    if not python.exists():
        print(f"building reference {ref} ({sha[:8]}) in {target} ...", flush=True)
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        archive = subprocess.Popen(
            ["git", "archive", sha], cwd=common.root(), stdout=subprocess.PIPE
        )
        subprocess.run(
            ["tar", "-x", "-C", str(target)], stdin=archive.stdout, check=True
        )
        if archive.wait():
            raise InfraError(f"git archive {sha} failed")
        sync = subprocess.run(
            ["uv", "sync", "--frozen", "--no-dev", "-q"],
            cwd=target,
            env=common.child_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if sync.returncode:
            shutil.rmtree(target, ignore_errors=True)
            raise InfraError(f"uv sync of {ref} failed: {sync.stderr.strip()[-400:]}")
    return Side(f"{ref}@{sha[:8]}", python)


def measure(side: Side, case: Case, args, mode: str = "bench") -> dict:
    """Run the worker once and return its result, or `{"error": ...}`."""
    cmd = [
        str(side.python),
        str(common.WORKER),
        "--circuit",
        str(case.path),
        "--mode",
        mode,
        "--sampler",
        case.sampler,
        "--postselect",
        case.postselect,
        "--min-seconds",
        str(args.min_seconds),
    ]
    if args.strategy:
        cmd += ["--strategy", args.strategy]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.bit_packed:
        cmd.append("--bit-packed")
    wrapped = common.capped(
        cmd, mem=args.mem, timeout=args.timeout, pin=common.pin_cpus(args.threads)
    )
    done = subprocess.run(
        wrapped,
        env=common.child_env(args.threads),
        capture_output=True,
        text=True,
        check=False,
    )
    for line in done.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    reason = common.explain_exit(done.returncode, args.mem, args.timeout)
    last = (done.stderr.strip().splitlines() or [""])[-1]
    return {"error": f"{reason}: {last}"[:200]}


def compare(new: list[float], ref: list[float], tolerance: float) -> dict:
    """Compare paired rounds of a higher-is-better quantity.

    The verdict is "faster" or "slower" only when the geometric mean of the
    per-round ratios is outside the tolerance and every round agrees on the sign;
    "noisy" when the mean is outside but the rounds disagree; otherwise "same".
    """
    ratios = [a / b for a, b in zip(new, ref, strict=True)]
    mean = math.exp(statistics.fmean(math.log(r) for r in ratios))
    low, high = min(ratios), max(ratios)
    if mean > 1 + tolerance:
        verdict = "faster" if low > 1 else "noisy"
    elif mean < 1 - tolerance:
        verdict = "slower" if high < 1 else "noisy"
    else:
        verdict = "same"
    return {"ratio": mean, "low": low, "high": high, "verdict": verdict}


def run_case(case: Case, sides: list[Side], args) -> dict:
    """Prime the caches, then measure every side in interleaved rounds."""
    rounds: dict[str, list[dict]] = {side.label: [] for side in sides}
    record: dict = {"circuit": case.name, "path": str(case.path)}
    for side in sides:
        primed = measure(side, case, args, mode="prime")
        if "error" in primed:
            record["error"] = f"{side.label}: {primed['error']}"
            return record
    for index in range(args.rounds):
        order = sides if index % 2 == 0 else sides[::-1]
        for side in order:
            result = measure(side, case, args)
            if "error" in result:
                record["error"] = f"{side.label}: {result['error']}"
                return record
            rounds[side.label].append(result)
    record["rounds"] = rounds
    new = rounds[sides[0].label]
    record["k"] = new[0].get("active_width")
    record["shots_per_s"] = statistics.median(r["shots_per_s"] for r in new)
    record["compile_s"] = statistics.median(r["compile_s"] for r in new)
    rates = [r["shots_per_s"] for r in new]
    record["spread"] = (max(rates) - min(rates)) / record["shots_per_s"]
    if len(sides) == 2:
        old = rounds[sides[1].label]
        record["ref_shots_per_s"] = statistics.median(r["shots_per_s"] for r in old)
        record["ref_compile_s"] = statistics.median(r["compile_s"] for r in old)
        record["rate"] = compare(rates, [r["shots_per_s"] for r in old], args.tolerance)
        # NOTE: compile time is lower-is-better, so the ratio is ref over new. Small
        # absolute differences are ignored: they are import and cache-load noise.
        record["compile"] = compare(
            [r["compile_s"] for r in old],
            [r["compile_s"] for r in new],
            max(args.tolerance, 0.10),
        )
        if abs(record["compile_s"] - record["ref_compile_s"]) < 0.05:
            record["compile"]["verdict"] = "same"
    return record


def format_row(record: dict, has_ref: bool) -> str:
    """Format one result line."""
    name = record["circuit"][:38]
    if "error" in record:
        return f"{name:<38}  ERROR: {record['error']}"
    k = "-" if record["k"] is None else record["k"]
    row = f"{name:<38} {k!s:>3} {common.si(record['shots_per_s']):>8}"
    row += f" {record['compile_s']:>8.2f}"
    if not has_ref:
        return row + f"   ±{record['spread'] * 50:.0f}%"
    rate, comp = record["rate"], record["compile"]
    row += f"   {rate['ratio']:.2f}x [{rate['low']:.2f}, {rate['high']:.2f}]"
    row += f" {rate['verdict']:<6}  {comp['ratio']:.2f}x {comp['verdict']}"
    return row


def run(args) -> int:
    """Run the benchmark and return the exit code."""
    try:
        cases = select(args)
        sides = [Side(common.describe(common.root()), Path(sys.executable))]
        if args.ref:
            sides.append(reference(args.ref))
    except InfraError as error:
        common.die(str(error))
    has_ref = len(sides) == 2
    rounds = args.rounds or (5 if has_ref else 3)
    args.rounds = rounds
    header = f"bench: new={sides[0].label}"
    if has_ref:
        header += f" ref={sides[1].label}"
    busy = common.busy_cores()
    header += f" threads={args.threads} rounds={rounds} busy_cores={busy:.1f}"
    print(header)
    if busy > 1.0:
        print("WARNING: other processes are using the CPU; expect noisy numbers")
    columns = f"{'circuit':<38} {'k':>3} {'shots/s':>8} {'compile':>8}"
    if has_ref:
        columns += "   shots/s vs ref (min, max)      compile vs ref"
    print(columns, flush=True)

    records = []
    for case in cases:
        record = run_case(case, sides, args)
        records.append(record)
        print(format_row(record, has_ref), flush=True)

    out = common.out_dir("bench") / f"{common.stamp()}.json"
    meta = {
        "new": sides[0].label,
        "ref": sides[1].label if has_ref else None,
        "threads": args.threads,
        "rounds": rounds,
        "tolerance": args.tolerance,
        "busy_cores": busy,
        "cpu_count": os.cpu_count(),
    }
    out.write_text(json.dumps({"meta": meta, "results": records}, indent=1))

    errors = sum("error" in record for record in records)
    verdicts = [
        record[key]["verdict"]
        for record in records
        if "error" not in record and has_ref
        for key in ("rate", "compile")
    ]
    slower = verdicts.count("slower")
    summary = f"SUMMARY: {len(records)} circuits, {errors} errors"
    if has_ref:
        summary += (
            f", {slower} slower, {verdicts.count('faster')} faster, "
            f"{verdicts.count('noisy')} noisy (tolerance {args.tolerance:.0%})"
        )
    print(f"{summary}. Details: {out}")
    if errors:
        return INFRA
    return FAILED if slower else OK


def fetch_corpus(_args) -> int:
    """Clone or update the tsim-bench corpus in the shared cache."""
    target = common.cache_dir() / "tsim-bench"
    if (target / ".git").is_dir():
        cmd = ["git", "-C", str(target), "pull", "--ff-only", "-q"]
    else:
        url = "https://github.com/rafaelha/tsim-bench"
        cmd = ["git", "clone", "--depth", "1", "-q", url, str(target)]
    if subprocess.run(cmd, check=False).returncode:
        common.die("could not fetch the tsim-bench corpus")
    count = len(corpus_cases(target, 10**6))
    print(f"corpus at {target}: {count} circuits with a recorded active width")
    return OK
