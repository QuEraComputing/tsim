"""Shared helpers: paths, the environment of child processes, and memory caps."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

DEV_DIR = Path(__file__).resolve().parent.parent

# Exit codes shared by every subcommand.
OK, FAILED, INFRA, RUNNING = 0, 1, 2, 3


class InfraError(Exception):
    """The harness could not run the request (as opposed to the code failing it)."""


def die(message: str) -> NoReturn:
    """Print one greppable error line and exit with the infrastructure code."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(INFRA)


def git(*args: str, cwd: Path | None = None) -> str:
    """Run git and return its stripped stdout."""
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise InfraError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


@functools.cache
def root() -> Path:
    """Return the checkout that the command was started in."""
    try:
        return Path(git("rev-parse", "--show-toplevel"))
    except InfraError:
        return DEV_DIR.parent


def out_dir(kind: str) -> Path:
    """Return (and create) a directory for logs and results inside the checkout."""
    path = root() / ".tsim-dev" / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    """Return the cache shared by every checkout (reference builds, corpus)."""
    path = Path(os.environ.get("TSIM_DEV_CACHE", "~/.cache/tsim-dev")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def stamp() -> str:
    """Return a timestamp that sorts and is safe in file names."""
    return time.strftime("%Y%m%d-%H%M%S")


def describe(path: Path) -> str:
    """Describe a checkout as branch@commit, with -dirty when it has changes."""
    try:
        branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=path)
        commit = git("rev-parse", "--short=8", "HEAD", cwd=path)
        if git("status", "--porcelain", "--untracked-files=no", cwd=path):
            commit += "-dirty"
    except InfraError:
        return str(path)
    return f"{branch}@{commit}"


def child_env(threads: int | None = None) -> dict[str, str]:
    """Return the environment for Tsim processes.

    JAX is kept on the CPU unless the caller already chose a platform, which also
    silences its missing-CUDA warning. The thread variables are set when given.
    """
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.pop("VIRTUAL_ENV", None)
    if threads is not None:
        for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = str(threads)
    return env


@functools.cache
def has_systemd() -> bool:
    """Return whether transient user scopes work on this machine."""
    if not shutil.which("systemd-run"):
        return False
    probe = subprocess.run(
        ["systemd-run", "--user", "--scope", "-q", "true"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def capped(
    cmd: list[str],
    *,
    mem: str,
    cpus: float | None = None,
    timeout: int | None = None,
    pin: list[int] | None = None,
) -> list[str]:
    """Wrap a command in a memory-capped scope, optionally pinned and timed out.

    Without systemd (macOS, some containers) the command runs uncapped, with a
    warning, so that the CLI still works there.
    """
    if pin and shutil.which("taskset"):
        cmd = ["taskset", "-c", ",".join(map(str, pin)), *cmd]
    if timeout and shutil.which("timeout"):
        cmd = ["timeout", "--signal=TERM", "--kill-after=10", str(timeout), *cmd]
    if not has_systemd():
        print("WARNING: no systemd user session, running without a memory cap")
        return cmd
    props = ["-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0"]
    if cpus:
        props += ["-p", f"CPUQuota={round(cpus * 100)}%"]
    return ["systemd-run", "--user", "--scope", "-q", *props, *cmd]


def explain_exit(code: int, mem: str, timeout: int | None) -> str:
    """Name the usual reasons for a capped command's exit code."""
    if code == 124:
        return f"timed out after {timeout} s"
    if code in (137, -9):
        return f"killed, most likely by the {mem} memory cap"
    return f"exit code {code}"


def pin_cpus(threads: int) -> list[int]:
    """Pick the last `threads` CPUs this process may use."""
    # NOTE: a `sys.platform` check, unlike `hasattr`, lets pyright on macOS accept
    # this Linux-only call.
    if sys.platform != "linux":
        return []
    cpus = sorted(os.sched_getaffinity(0))
    return cpus[-threads:]


def busy_cores(interval: float = 0.3) -> float:
    """Return how many cores other processes keep busy right now."""

    def snapshot() -> tuple[int, int]:
        fields = [
            int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]
        ]
        return sum(fields), fields[3] + fields[4]

    try:
        total0, idle0 = snapshot()
        time.sleep(interval)
        total1, idle1 = snapshot()
    except (OSError, ValueError, IndexError):
        return os.getloadavg()[0]
    if total1 == total0:
        return 0.0
    return (1 - (idle1 - idle0) / (total1 - total0)) * (os.cpu_count() or 1)


def si(value: float) -> str:
    """Format a count or rate the short way: 950, 3.2k, 2.1M."""
    for bound, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= bound:
            return f"{value / bound:.3g}{suffix}"
    return f"{value:.3g}"


def tail(path: Path, lines: int) -> str:
    """Return the last lines of a text file."""
    if not path.exists():
        return ""
    text = path.read_text(errors="replace").splitlines()
    return "\n".join(text[-lines:])
