"""`tsim-dev inspect` and `tsim-dev profile`: what a circuit costs, and where."""

from __future__ import annotations

import collections
import json
import re
import subprocess
import sys
from pathlib import Path

from . import bench, common
from .common import INFRA, OK, InfraError

PHASES = ("phase_import", "phase_compile", "phase_sample")
UNKNOWN = re.compile(r"0x[0-9a-f]+ \(\?\)")
JITTED = "[jitted numba code]"


def _case(args) -> bench.Case:
    args.only, args.suite, args.max_width, args.width_of = "", "quick", None, "tsim"
    args.circuits = [args.circuit]
    return bench.select(args)[0]


def _defaults(args) -> None:
    for name, value in (("strategy", ""), ("batch_size", 0), ("bit_packed", False)):
        if not hasattr(args, name):
            setattr(args, name, value)


def inspect(args) -> int:
    """Print the size and active width of a circuit, without sampling it."""
    _defaults(args)
    args.min_seconds = 0.0
    try:
        case = _case(args)
    except InfraError as error:
        common.die(str(error))
    side = bench.Side("new", Path(sys.executable))
    result = bench.measure(side, case, args, mode="inspect")
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return INFRA
    print(f"{case.name}: {result['sampler']} sampler, Tsim {result['tsim_version']}")
    for key in ("qubits", "tcount", "measurements", "detectors", "active_width"):
        print(f"  {key}: {result[key]}")
    print(f"  compile_s: {result['compile_s']:.2f}")
    if result.get("stats"):
        print("  stats: " + result["stats"].replace("\n", "\n    "))
    return OK


def fold(path: Path) -> dict[str, collections.Counter]:
    """Split py-spy's folded stacks by phase and count samples per frame.

    Returns, for each phase, counters keyed by ("self", frame) for the innermost
    frame, ("leaf", frame) for the innermost Tsim frame, and ("total", frame) for
    every Tsim frame on the stack. Jitted numba code has no symbols, so its
    addresses are counted together.
    """
    phases: dict[str, collections.Counter] = {p: collections.Counter() for p in PHASES}
    for line in path.read_text(errors="replace").splitlines():
        stack, _, count = line.rpartition(" ")
        if not count.isdigit():
            continue
        phase = next((p for p in PHASES if p in stack), None)
        if phase is None:
            continue
        frames = [re.sub(r":\d+\)$", ")", frame) for frame in stack.split(";")]
        frames = [JITTED if UNKNOWN.fullmatch(f) else f for f in frames]
        ours = [f for f in frames if "tsim/" in f]
        counter = phases[phase]
        counter["samples", ""] += int(count)
        counter["self", frames[-1]] += int(count)
        # NOTE: kernels.py only dispatches jitted kernels, so the frame that called
        # it says more about where the time goes.
        callers = [f for f in ours if "tsim/utils/kernels.py" not in f]
        if callers:
            counter["leaf", callers[-1]] += int(count)
        for frame in set(ours):
            counter["total", frame] += int(count)
    return phases


def profile(args) -> int:
    """Sample the stacks of one run with py-spy and print the top frames per phase."""
    _defaults(args)
    spy = Path(sys.executable).parent / "py-spy"
    if not spy.exists():
        common.die("py-spy is not installed in this environment; run `uv sync`")
    try:
        case = _case(args)
    except InfraError as error:
        common.die(str(error))
    side = bench.Side("new", Path(sys.executable))
    args.min_seconds = args.seconds / 10
    primed = bench.measure(side, case, args, mode="prime")
    if "error" in primed:
        print(f"ERROR: {primed['error']}")
        return INFRA

    raw = (
        common.out_dir("profile")
        / f"{case.name.replace('/', '_')}-{common.stamp()}.folded"
    )
    worker = bench.worker_command(Path(sys.executable), case, args, "profile")
    spy_cmd = [str(spy), "record", "--native", "-f", "raw", "-r", str(args.rate)]
    cmd = common.capped(
        [*spy_cmd, "-o", str(raw), "--", *worker],
        mem=args.mem,
        timeout=args.timeout,
        pin=common.pin_cpus(args.threads),
    )
    done = subprocess.run(
        cmd,
        env=common.child_env(args.threads),
        capture_output=True,
        text=True,
        check=False,
    )
    result = next(
        (
            json.loads(x[7:])
            for x in done.stdout.splitlines()
            if x.startswith("RESULT ")
        ),
        None,
    )
    if result is None or not raw.exists():
        reason = common.explain_exit(done.returncode, args.mem, args.timeout)
        print(f"ERROR: profile failed ({reason}): {done.stderr.strip()[-300:]}")
        return INFRA

    print(
        f"profile {case.name}: k={result['active_width']} threads={args.threads} "
        f"import {result['import_s']:.2f} s, compile {result['compile_s']:.2f} s "
        f"(warm numba cache), sampling {args.seconds:g} s at {result['shots']} shots/call"
    )
    for phase, counter in fold(raw).items():
        total = counter["samples", ""]
        if phase == "phase_import" or not total:
            continue
        print(f"\n{phase[6:]}: {total} samples")
        tables = (
            ("leaf", "innermost Tsim frame (kernel callers)"),
            ("self", "innermost frame"),
            ("total", "Tsim frames, inclusive"),
        )
        for kind, title in tables:
            rows = [(n, f) for (k, f), n in counter.items() if k == kind]
            print(f"  {title}:")
            for count, frame in sorted(rows, reverse=True)[: args.top]:
                print(f"    {count / total:6.1%}  {frame[:110]}")
    print(f"\nFolded stacks (speedscope, flamegraph.pl): {raw}")
    return OK
