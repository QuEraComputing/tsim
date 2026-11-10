"""The `tsim-dev` command line.

Exit codes: 0 success, 1 the code under test failed (tests, lint, a slower
benchmark, a failed job), 2 the harness could not run, 3 a job is still running.
"""

from __future__ import annotations

import argparse
import sys

from . import bench, checks, jobs, profile


def _caps(parser, mem: str, timeout: int, cpus: float | None = None) -> None:
    parser.add_argument("--mem", default=mem, help=f"memory cap (default {mem})")
    parser.add_argument(
        "--timeout", type=int, default=timeout, help=f"seconds (default {timeout})"
    )
    if cpus is not None:
        parser.add_argument(
            "--cpus", type=float, default=cpus, help=f"CPU quota (default {cpus:g})"
        )


def _corpus_dir(parser) -> None:
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="local tsim-bench checkout (default: TSIM_BENCH_DIR, else the pinned commit)",
    )


def _selection(parser, suite: str) -> None:
    parser.add_argument(
        "--suite",
        default=suite,
        help=f"tsim-bench suite, or `corpus` for every circuit (default {suite})",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="skip circuits whose recorded active width is larger",
    )
    parser.add_argument(
        "--width-of",
        default="tsim",
        help="backend whose recorded width --max-width uses (default tsim)",
    )
    parser.add_argument(
        "--only", default="", help="keep circuits whose id contains this"
    )
    _corpus_dir(parser)


def _circuit_options(parser) -> None:
    parser.add_argument(
        "--no-postselect",
        action="store_true",
        help="ignore the postselection recorded in tsim-bench",
    )
    parser.add_argument(
        "--sampler",
        default="auto",
        choices=["auto", "detector", "measurement"],
        help="for a .stim path outside tsim-bench",
    )
    parser.add_argument(
        "--postselect",
        default="none",
        choices=["none", "all"],
        help="for a .stim path outside tsim-bench",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="numba threads (default 1)"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="tsim-dev", description=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)

    p = sub.add_parser("check", help="gate for a finished change: lint, then all tests")
    p.add_argument("--no-tests", action="store_true", help="lint only")
    _caps(p, "12G", 1800)
    p.set_defaults(func=checks.check)

    p = sub.add_parser("test", help="run tests with a short summary")
    p.add_argument("--changed", action="store_true", help="only tests of changed files")
    p.add_argument("pytest_args", nargs="*", help="paths; pytest options go after --")
    _caps(p, "12G", 1800)
    p.set_defaults(func=checks.test)

    p = sub.add_parser("bench", help="throughput and compile time, optionally vs a ref")
    p.add_argument(
        "circuits", nargs="*", help="tsim-bench ids or file stems, or .stim paths"
    )
    p.add_argument("--ref", help="git revision or checkout path to compare against")
    p.add_argument(
        "--side",
        action="append",
        default=[],
        metavar="REF[:KEY=VALUE,...]",
        help="a side with its own settings, instead of --ref; give two to compare. "
        "REF: `.`, a git revision, a checkout or a python executable. "
        "Keys: threads=N, device=cpu|gpu, strategy=NAME. "
        "Example: --side .:threads=10 --side .:threads=1",
    )
    _selection(p, "quick")
    p.add_argument("--rounds", type=int, default=0, help="default 5 with --ref, else 3")
    p.add_argument(
        "--min-seconds", type=float, default=0.5, help="length of a timed call"
    )
    p.add_argument("--tolerance", type=float, default=0.05, help="default 0.05")
    p.add_argument("--strategy", default="")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--bit-packed", action="store_true")
    _circuit_options(p)
    _caps(p, "8G", 600)
    p.set_defaults(func=bench.run)

    p = sub.add_parser(
        "corpus", help="fetch the pinned tsim-bench, or list its circuits"
    )
    p.add_argument("action", choices=["fetch", "list"])
    _selection(p, "corpus")
    p.set_defaults(func=bench.corpus_command)

    p = sub.add_parser("inspect", help="size and active width of a circuit")
    p.add_argument("circuit", help="tsim-bench id or file stem, or a .stim path")
    _circuit_options(p)
    _corpus_dir(p)
    _caps(p, "8G", 600)
    p.set_defaults(func=profile.inspect)

    p = sub.add_parser("profile", help="where compile and sampling time goes (py-spy)")
    p.add_argument("circuit")
    p.add_argument(
        "--seconds", type=float, default=5.0, help="sampling time to profile"
    )
    p.add_argument("--rate", type=int, default=500, help="samples per second")
    p.add_argument("--top", type=int, default=12, help="frames per table")
    _circuit_options(p)
    _corpus_dir(p)
    _caps(p, "8G", 600)
    p.set_defaults(func=profile.profile)

    p = sub.add_parser(
        "run", help="run `-- command ...` under memory, CPU and time caps"
    )
    _caps(p, "8G", 1800, cpus=4)
    p.set_defaults(func=jobs.run)

    job = sub.add_parser("job", help="capped background jobs")
    job_sub = job.add_subparsers(dest="job_command", required=True)
    p = job_sub.add_parser("start", help="start `NAME -- command ...` as a job")
    p.add_argument("name")
    _caps(p, "8G", 3600, cpus=4)
    p.set_defaults(func=jobs.start)
    p = job_sub.add_parser(
        "status", help="state and last log lines of a job, or all jobs"
    )
    p.add_argument("name", nargs="?")
    p.add_argument("--lines", type=int, default=15)
    p.set_defaults(func=jobs.status)
    p = job_sub.add_parser(
        "wait", help="block until a job ends (exit 3 if still running)"
    )
    p.add_argument("name")
    p.add_argument(
        "--timeout", type=int, default=540, help="seconds to wait (default 540)"
    )
    p.add_argument("--lines", type=int, default=15)
    p.set_defaults(func=jobs.wait)
    p = job_sub.add_parser("stop", help="stop a job and its children")
    p.add_argument("name")
    p.set_defaults(func=jobs.stop)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the command line."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # NOTE: everything after "--" is the command to run (or options for pytest), so
    # it is split off before argparse can read the command's own options.
    rest: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, rest = argv[:split], argv[split + 1 :]
    args = build_parser().parse_args(argv)
    args.command = rest
    raise SystemExit(args.func(args))
