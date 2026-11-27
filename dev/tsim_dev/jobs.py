"""`tsim-dev run` and `tsim-dev job`: commands under hard memory and time caps.

A job is a transient systemd user unit, so it can be watched, waited for and
stopped by name. That replaces `nohup ... &`, sleep loops and `pkill -f`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time

from . import common
from .common import FAILED, INFRA, OK, RUNNING


def run(args) -> int:
    """Run a command in the foreground under the caps."""
    if not args.command:
        common.die(
            "give a command after `--`, for example: tsim-dev run -- python x.py"
        )
    cmd = common.capped(
        args.command, mem=args.mem, cpus=args.cpus, timeout=args.timeout
    )
    threads = max(1, int(args.cpus)) if args.cpus else None
    code = subprocess.run(cmd, env=common.child_env(threads), check=False).returncode
    if code:
        # NOTE: the documented codes are 0 pass and 1 failed; the command's own
        # exit code is printed, not passed on.
        reason = common.explain_exit(code, args.mem, args.timeout)
        print(f"FAILED: {reason}", flush=True)
    return FAILED if code else OK


def _unit(name: str) -> str:
    return f"tsim-dev-{name}"


def _paths(name: str):
    base = common.out_dir("jobs")
    return base / f"{name}.json", base / f"{name}.log", base / f"{name}.exit"


def _show(name: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            _unit(name),
            "-p",
            "ActiveState,Result,MemoryPeak,MemoryCurrent",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def _state(name: str) -> tuple[str, int]:
    """Return a one-line state and the exit code that describes it."""
    meta, _, exit_file = _paths(name)
    if not meta.exists():
        return "unknown job", INFRA
    show = _show(name)
    started = json.loads(meta.read_text())["started"]
    age = f"{time.time() - started:.0f} s"
    if show.get("ActiveState") in ("active", "activating", "deactivating"):
        peak = show.get("MemoryPeak", "")
        memory = f", peak memory {int(peak) / 2**30:.1f} GB" if peak.isdigit() else ""
        return f"running for {age}{memory}", RUNNING
    if exit_file.exists() and exit_file.read_text().strip().isdigit():
        code = int(exit_file.read_text())
        if code == 0:
            return "finished, exit code 0", OK
        if code == 137:
            return "failed: killed, most likely by the memory cap", FAILED
        return f"failed, exit code {code}", FAILED
    reasons = {"oom-kill": "killed by the memory cap", "timeout": "hit the time limit"}
    return reasons.get(show.get("Result", ""), "stopped before it finished"), FAILED


def start(args) -> int:
    """Start a command as a named background job."""
    if not common.has_systemd():
        common.die("`job` needs a systemd user session; use `tsim-dev run` instead")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        common.die("job names may contain letters, digits, '.', '_' and '-'")
    if not args.command:
        common.die("give a command after `--`")
    if _state(args.name)[1] == RUNNING:
        common.die(f"job {args.name} is already running; stop it or pick another name")
    meta, log, exit_file = _paths(args.name)
    exit_file.unlink(missing_ok=True)
    log.write_text("")
    subprocess.run(
        ["systemctl", "--user", "reset-failed", _unit(args.name)],
        capture_output=True,
        check=False,
    )
    env = common.child_env(max(1, int(args.cpus)))
    env["PYTHONUNBUFFERED"] = "1"
    setenv = [
        f"--setenv={key}={value}"
        for key, value in env.items()
        if "\n" not in value and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
    ]
    # NOTE: the shell records the exit code, so a finished job can be told apart
    # from one that systemd killed for its memory or time limit.
    script = '"$@"; code=$?; echo $code > "$TSIM_DEV_EXIT"; exit $code'
    cmd = [
        "systemd-run",
        "--user",
        "-q",
        f"--unit={_unit(args.name)}",
        "--same-dir",
        "-p",
        f"MemoryMax={args.mem}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        f"CPUQuota={round(args.cpus * 100)}%",
        "-p",
        f"RuntimeMaxSec={args.timeout}",
        "-p",
        f"StandardOutput=append:{log}",
        "-p",
        "StandardError=inherit",
        f"--setenv=TSIM_DEV_EXIT={exit_file}",
        *setenv,
        "bash",
        "-c",
        script,
        "job",
        *args.command,
    ]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode:
        common.die(f"systemd-run failed: {done.stderr.strip()[-300:]}")
    meta.write_text(
        json.dumps(
            {"command": args.command, "started": time.time(), "cwd": os.getcwd()}
        )
    )
    print(
        f"started job {args.name} (mem {args.mem}, cpus {args.cpus:g}, "
        f"limit {args.timeout} s). Log: {log}\n"
        f"Next: tsim-dev job wait {args.name}"
    )
    return OK


def _report(name: str, lines: int) -> int:
    state, code = _state(name)
    print(f"job {name}: {state}")
    if code != INFRA and lines:
        text = common.tail(_paths(name)[1], lines)
        if text:
            print(text)
    return code


def status(args) -> int:
    """Report one job, or list every job of this checkout."""
    if args.name:
        return _report(args.name, args.lines)
    names = sorted(path.stem for path in common.out_dir("jobs").glob("*.json"))
    for name in names:
        print(f"job {name}: {_state(name)[0]}")
    if not names:
        print("no jobs")
    return OK


def wait(args) -> int:
    """Block until the job ends or the wait times out, then report it."""
    deadline = time.time() + args.timeout
    while _state(args.name)[1] == RUNNING and time.time() < deadline:
        time.sleep(2)
    return _report(args.name, args.lines)


def stop(args) -> int:
    """Stop a job and everything it started."""
    subprocess.run(
        ["systemctl", "--user", "stop", _unit(args.name)],
        capture_output=True,
        check=False,
    )
    print(f"job {args.name}: {_state(args.name)[0]}")
    return OK
