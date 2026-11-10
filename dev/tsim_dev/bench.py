"""`tsim-dev bench`: throughput and compile time, optionally against a reference.

Every measurement runs in a fresh process under a memory cap, pinned to fixed
cores, with numba's on-disk cache warm. With `--ref`, the two checkouts are
measured in interleaved rounds and compared round by round.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import common, corpus, width
from .common import FAILED, INFRA, OK, InfraError


@dataclass(frozen=True)
class Case:
    """One circuit with the sampler options it is measured with.

    A corpus circuit (`id` set) is postselected as its tsim-bench manifest says, unless
    `postselect` is "none". A `.stim` path outside the corpus uses `sampler` and
    `postselect` ("all" or "none") from the command line.
    """

    name: str
    path: Path
    sampler: str = "auto"
    postselect: str = "none"  # "recorded", "all" or "none"
    id: str | None = None
    width: int | None = None  # recorded width of the --width-of backend
    tsim_width: int | None = None  # recorded Tsim width


@dataclass(frozen=True)
class Side:
    """One side of a comparison: the Python that imports its Tsim, and its settings.

    `threads` and `strategy` override the command line's when set.
    """

    label: str
    python: Path
    threads: int | None = None
    strategy: str | None = None

    def describe(self) -> str:
        """Return the label with the settings that this side overrides."""
        extra = [f"threads={self.threads}"] if self.threads is not None else []
        extra += [f"strategy={self.strategy}"] if self.strategy is not None else []
        return " ".join([self.label, *extra])


SIDE_KEYS = ("threads", "strategy")


def parse_side(spec: str) -> Side:
    """Parse `--side REF[:key=value,...]`.

    REF is `.` (this checkout), a git revision, the path of another checkout with a
    `.venv`, or the path of a Python executable (an environment built elsewhere, such
    as one with other dependencies). Keys: threads=N, strategy=NAME.
    """
    ref, _, options = spec.partition(":")
    settings: dict = {}
    for item in filter(None, options.split(",")):
        key, sep, value = item.partition("=")
        if not sep or key not in SIDE_KEYS:
            raise InfraError(
                f"bad --side option {item!r}: use "
                + ", ".join(f"{k}=..." for k in SIDE_KEYS)
            )
        settings[key] = value
    if "threads" in settings:
        if not settings["threads"].isdigit() or int(settings["threads"]) < 1:
            raise InfraError(f"bad thread count in --side {spec!r}")
        settings["threads"] = int(settings["threads"])
    if ref in ("", "."):
        base = Side(common.describe(common.root()), Path(sys.executable))
    elif Path(ref).expanduser().is_file():
        # NOTE: not resolved: a venv's python is a symlink to the base interpreter,
        # and resolving it would lose the environment.
        python = Path(ref).expanduser().absolute()
        base = Side(str(python), python)
    else:
        base = reference(ref)
    return Side(base.label, base.python, **settings)


def _from_corpus(entry, args) -> Case:
    postselect = (
        "none" if args.no_postselect or entry.postselect == "none" else "recorded"
    )
    return Case(
        entry.id.split("/", 1)[1],
        entry.path,
        entry.sampler,
        postselect,
        entry.id,
        entry.width(args.width_of),
        entry.width("tsim"),
    )


def _lookup(item: str, everything: dict) -> object | None:
    """Return the corpus entry with this id or path, or the only one with this file stem."""
    if item in everything:
        return everything[item]
    if item.endswith(".stim") and Path(item).is_file():
        path = Path(item).resolve()
        return next((e for e in everything.values() if e.path.resolve() == path), None)
    matches = [entry for key, entry in everything.items() if key.endswith("/" + item)]
    if len(matches) > 1:
        raise InfraError(
            f"{item!r} is ambiguous: " + ", ".join(entry.id for entry in matches)
        )
    return matches[0] if matches else None


def select(args) -> list[Case]:
    """Turn the command line into the list of cases to measure."""
    if args.suite is None:  # compiling only is fast enough for the whole corpus
        args.suite = "corpus" if getattr(args, "width", False) else "quick"
    try:
        return _select(args)
    except (ValueError, RuntimeError) as error:  # a malformed tsim-bench manifest
        raise InfraError(f"tsim-bench: {error}") from None


def _select(args) -> list[Case]:
    bench_pkg = corpus.package(args.corpus_dir)
    if args.circuits:
        everything = bench_pkg.load(corpus.root(args.corpus_dir))
        cases = []
        for item in args.circuits:
            entry = _lookup(item, everything)
            if entry is not None:
                cases.append(_from_corpus(entry, args))
            elif Path(item).is_file():
                path = Path(item).resolve()
                postselect = "none" if args.no_postselect else args.postselect
                cases.append(Case(path.stem, path, args.sampler, postselect))
            else:
                raise InfraError(
                    f"unknown circuit {item!r}: give a .stim path or a tsim-bench id "
                    "(`tsim-dev corpus list`)"
                )
    else:
        try:
            entries = bench_pkg.cases(
                args.suite, width_of=args.width_of, root=corpus.root(args.corpus_dir)
            )
        except KeyError as error:
            raise InfraError(str(error.args[0])) from None
        cases = [_from_corpus(entry, args) for entry in entries]
    if args.max_width is not None:
        cases = [
            case
            for case in cases
            if case.id is None
            or (case.width is not None and case.width <= args.max_width)
        ]
    if args.only:
        cases = [case for case in cases if args.only in (case.id or case.name)]
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


def worker_command(python: Path, case: Case, args, mode: str) -> list[str]:
    """Return the command that times `case` with tsim-bench's worker."""
    cmd = [str(python), str(corpus.worker(args.corpus_dir)), "--mode", mode]
    if case.id is not None:
        cmd += ["--root", str(corpus.root(args.corpus_dir)), "--case", case.id]
        if case.postselect == "none":
            cmd.append("--no-postselect")
    else:
        cmd += ["--circuit", str(case.path), "--sampler", case.sampler]
        cmd += ["--postselect", case.postselect]
    cmd += ["--min-seconds", str(args.min_seconds)]
    if args.strategy:
        cmd += ["--strategy", args.strategy]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.bit_packed:
        cmd.append("--bit-packed")
    return cmd


def side_env(side: Side, threads: int) -> dict[str, str]:
    """Return the environment of one side's worker."""
    env = common.child_env(threads)
    return env


def measure(side: Side, case: Case, args, mode: str = "bench") -> dict:
    """Run the worker once and return its result, or `{"error": ...}`."""
    threads = side.threads or args.threads
    options = args
    if side.strategy is not None:
        options = argparse.Namespace(**{**vars(args), "strategy": side.strategy})
    cmd = worker_command(side.python, case, options, mode)
    wrapped = common.capped(
        cmd, mem=args.mem, timeout=args.timeout, pin=common.pin_cpus(threads)
    )
    done = subprocess.run(
        wrapped,
        env=side_env(side, threads),
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
    rounds: dict[str, list[dict]] = {side.describe(): [] for side in sides}
    record: dict = {
        "circuit": case.name,
        "id": case.id,
        "path": str(case.path),
        "postselect": case.postselect,
    }
    for side in sides:
        primed = measure(side, case, args, mode="prime")
        if "error" in primed:
            record["error"] = f"{side.describe()}: {primed['error']}"
            return record
    for index in range(args.rounds):
        order = sides if index % 2 == 0 else sides[::-1]
        for side in order:
            result = measure(side, case, args)
            if "error" in result:
                record["error"] = f"{side.describe()}: {result['error']}"
                return record
            rounds[side.describe()].append(result)
    record["rounds"] = rounds
    # NOTE: the width is fixed by the compiled sampler, so it must not change between
    # runs of one side; a change means compilation is not deterministic.
    for side in sides:
        widths = {r.get("active_width") for r in rounds[side.describe()]}
        if len(widths) > 1:
            record["error"] = (
                f"{side.describe()}: active width changed between runs: "
                + ", ".join(sorted(str(w) for w in widths))
            )
            return record
    new = rounds[sides[0].describe()]
    record["k"] = new[0].get("active_width")
    record["acceptance"] = new[0].get("acceptance")
    record["shots_per_s"] = statistics.median(r["shots_per_s"] for r in new)
    record["compile_s"] = statistics.median(r["compile_s"] for r in new)
    rates = [r["shots_per_s"] for r in new]
    record["spread"] = (max(rates) - min(rates)) / record["shots_per_s"]
    if len(sides) == 2:
        old = rounds[sides[1].describe()]
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
        record["ref_acceptance"] = old[0].get("acceptance")
        record["ref_k"] = old[0].get("active_width")
        record["width"] = width_verdict(record["k"], record["ref_k"])
        if not same_acceptance(new, old):
            # NOTE: a different kept fraction means the sides sample different
            # things, so their rates are not comparable.
            record["rate"]["verdict"] = KEPT_DIFFERS
    return record


KEPT_DIFFERS = "kept-differs"


def width_verdict(new: int | None, ref: int | None) -> str:
    """Compare the active widths of the two sides; lower is better."""
    if new is None or ref is None or new == ref:
        return "same"
    return "narrower" if new < ref else "wider"


def same_acceptance(new: list[dict], old: list[dict], sigmas: float = 6.0) -> bool:
    """Return whether two sides' kept fractions agree within binomial noise.

    Each round reports the kept fraction of `shots` requested shots per timed
    repeat; the count over all repeats is larger, so this test is conservative.
    """

    def pooled(rounds: list[dict]) -> tuple[float, float]:
        kept = sum(r["acceptance"] * r["shots"] for r in rounds)
        total = sum(r["shots"] for r in rounds)
        return kept / total, total

    if any(r.get("acceptance") is None for r in new + old):
        return True
    (p, n), (q, m) = pooled(new), pooled(old)
    both = (p * n + q * m) / (n + m)
    sigma = math.sqrt(max(both * (1 - both), 1e-12) * (1 / n + 1 / m))
    return abs(p - q) <= sigmas * sigma + 1e-9


NAME = 44


def format_row(record: dict, has_ref: bool) -> str:
    """Format one result line."""
    name = record["circuit"][:NAME]
    if "error" in record:
        return f"{name:<{NAME}}  ERROR: {record['error']}"
    k = "-" if record["k"] is None else record["k"]
    accepted = record.get("acceptance")
    kept = (
        "-" if record["postselect"] == "none" or accepted is None else f"{accepted:.3g}"
    )
    row = f"{name:<{NAME}} {k!s:>3} {common.si(record['shots_per_s']):>8} {kept:>6}"
    row += f" {record['compile_s']:>8.2f}"
    if not has_ref:
        return row + f"   ±{record['spread'] * 50:.0f}%"
    rate, comp = record["rate"], record["compile"]
    row += f"   {rate['ratio']:.2f}x [{rate['low']:.2f}, {rate['high']:.2f}]"
    row += f" {rate['verdict']:<6}  {comp['ratio']:.2f}x {comp['verdict']}"
    if record.get("width", "same") != "same":
        row += f"  {record['width'].upper()}: k {record['ref_k']} -> {record['k']}"
    if rate["verdict"] == KEPT_DIFFERS:
        row += (
            f"  (kept {record['acceptance']:.4g} vs ref {record['ref_acceptance']:.4g})"
        )
    return row


def run(args) -> int:
    """Run the benchmark and return the exit code."""
    try:
        if args.side and args.ref:
            raise InfraError("--side replaces --ref: give `--side . --side REF`")
        if len(args.side) > 2:
            raise InfraError("give at most two --side options")
        cases = select(args)
        if args.side:
            sides = [parse_side(spec) for spec in args.side]
        else:
            sides = [Side(common.describe(common.root()), Path(sys.executable))]
            if args.ref:
                sides.append(reference(args.ref))
        if len({side.describe() for side in sides}) < len(sides):
            raise InfraError("the two sides are the same; change a setting of one")
    except InfraError as error:
        common.die(str(error))
    if args.width:
        return width.run(args, cases, sides)
    has_ref = len(sides) == 2
    rounds = args.rounds or (5 if has_ref else 3)
    args.rounds = rounds
    header = f"bench: new={sides[0].describe()}"
    if has_ref:
        header += f" ref={sides[1].describe()}"
    busy = common.busy_cores()
    if not any(side.threads for side in sides):
        header += f" threads={args.threads}"
    header += f" rounds={rounds} busy_cores={busy:.1f}"
    header += f" postselect={'off' if args.no_postselect else 'on'}"
    header += f" tsim-bench={corpus.describe(args.corpus_dir)}"
    print(header)
    if busy > 1.0:
        print("WARNING: other processes are using the CPU; expect noisy numbers")
    columns = f"{'circuit':<{NAME}} {'k':>3} {'shots/s':>8} {'kept':>6} {'compile':>8}"
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
        "new": sides[0].describe(),
        "ref": sides[1].describe() if has_ref else None,
        "threads": [side.threads or args.threads for side in sides],
        "rounds": rounds,
        "tolerance": args.tolerance,
        "busy_cores": busy,
        "cpu_count": os.cpu_count(),
        "postselect": not args.no_postselect,
        "tsim_bench": corpus.describe(args.corpus_dir),
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
    differs = verdicts.count(KEPT_DIFFERS)
    widths = [r.get("width") for r in records if "error" not in r]
    wider, narrower = widths.count("wider"), widths.count("narrower")
    summary = f"SUMMARY: {len(records)} circuits, {errors} errors"
    if has_ref:
        summary += (
            f", {slower} slower, {verdicts.count('faster')} faster, "
            f"{verdicts.count('noisy')} noisy (tolerance {args.tolerance:.0%})"
        )
        if differs:
            summary += (
                f", {differs} with a different kept fraction (rates not comparable)"
            )
        if wider or narrower:
            summary += f", active width: {wider} wider, {narrower} narrower"
    print(f"{summary}. Details: {out}")
    if errors:
        return INFRA
    return FAILED if slower or differs or wider else OK


def corpus_command(args) -> int:
    """`tsim-dev corpus fetch|list`: fetch the pinned tsim-bench, or list circuits."""
    try:
        root = corpus.root(args.corpus_dir)
        bench_pkg = corpus.package(args.corpus_dir)
        entries = bench_pkg.cases(
            args.suite,
            max_width=args.max_width,
            width_of=args.width_of,
            only=args.only or None,
            root=root,
        )
    except (InfraError, KeyError, ValueError, RuntimeError) as error:
        common.die(str(error.args[0]))
    if args.action == "fetch":
        print(
            f"tsim-bench {corpus.describe(args.corpus_dir)} at {root}: {len(entries)} circuits"
        )
        return OK
    for entry in entries:
        masked = len(bench_pkg.mask_indices(entry))
        width = entry.width(args.width_of)
        print(
            f"{entry.id:<80} k={'-' if width is None else width!s:>3} "
            f"arXiv:{entry.arxiv:<10} postselect {masked}/{entry.num_detectors}"
        )
    print(f"{len(entries)} circuits")
    return OK
