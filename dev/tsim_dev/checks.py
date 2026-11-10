"""`tsim-dev test` and `tsim-dev check`: terse, capped test and lint runs."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from . import common
from .common import FAILED, OK, InfraError


def changed_files() -> list[str]:
    """Return files that differ from the merge base with main, plus untracked ones."""
    root = common.root()
    base = "HEAD"
    for candidate in ("origin/main", "main"):
        try:
            base = common.git("merge-base", "HEAD", candidate, cwd=root)
            break
        except InfraError:
            continue
    names = common.git("diff", "--name-only", base, cwd=root).splitlines()
    names += common.git(
        "ls-files", "--others", "--exclude-standard", cwd=root
    ).splitlines()
    return sorted({name for name in names if (root / name).exists()})


def tests_for(files: list[str]) -> list[str] | None:
    """Map changed files to test files; None means "run the full suite"."""
    root = common.root()
    tests: set[str] = set()
    for name in files:
        path = Path(name)
        if path.suffix != ".py" and path.parts[0] not in ("skills", "dev"):
            continue
        if path.parts[0] == "test":
            if path.name.startswith("test_"):
                tests.add(name)
            elif "helpers" in path.parts:
                return None
        elif path.parts[0] == "skills":
            tests.add("test/unit/test_skills.py")
        elif path.parts[0] == "dev":
            tests.add("test/unit/test_dev_cli.py")
        elif path.parts[:2] == ("src", "tsim"):
            inner = Path(*path.parts[2:-1])
            candidates = [
                Path("test/unit") / inner / f"test_{path.name}",
                Path("test/unit") / f"test_{path.name}",
                Path("test/integration") / f"test_{path.name}",
            ]
            found = [str(c) for c in candidates if (root / c).exists()]
            if not found:
                return None
            tests.update(found)
    return sorted(tests)


def summarize(log: Path, limit: int = 40) -> str:
    """Return the failures and the final line of a pytest log."""
    lines = log.read_text(errors="replace").splitlines()
    if not lines:
        return "(no output)"
    marker = next(
        (i for i, line in enumerate(lines) if "short test summary info" in line),
        len(lines) - 3,
    )
    out: list[str] = []
    first = next((i for i, line in enumerate(lines) if line.startswith("____")), None)
    if first is not None and first < marker:
        out += lines[first : min(first + limit, marker)]
    out += lines[marker:][-limit:]
    return "\n".join(out)


def run_pytest(targets: list[str], mem: str, timeout: int) -> tuple[int, Path]:
    """Run pytest under the caps with its output in a log file."""
    log = common.out_dir("logs") / f"test-{common.stamp()}.log"
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-rfE", "--tb=short"]
    wrapped = common.capped([*cmd, *targets], mem=mem, timeout=timeout)
    with open(log, "w") as file:
        code = subprocess.run(
            wrapped,
            cwd=common.root(),
            env=common.child_env(),
            stdout=file,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    return code, log


def test(args) -> int:
    """Run the tests of the changed files, the full suite, or the given targets."""
    targets = list(args.pytest_args)
    if not targets and args.changed:
        mapped = tests_for(changed_files())
        if mapped == []:
            print(
                "no changed files map to tests; use `tsim-dev test` for the full suite"
            )
            return OK
        if mapped is None:
            print("a changed file has no matching test file: running the full suite")
        targets = mapped or []
    targets = [*(targets or ["test"]), *args.command]
    start = time.time()
    code, log = run_pytest(targets, args.mem, args.timeout)
    print(summarize(log))
    if code in (124, 137, -9):
        print(f"ERROR: {common.explain_exit(code, args.mem, args.timeout)}")
    verdict = "PASS" if code == 0 else "FAIL"
    print(f"{verdict} tests ({time.time() - start:.0f} s). Log: {log}")
    return OK if code == 0 else FAILED


def check(args) -> int:
    """Run the gate for a finished change: pre-commit on changed files, then tests."""
    files = changed_files()
    failed = False

    start = time.time()
    log = common.out_dir("logs") / f"lint-{common.stamp()}.log"
    cmd = [sys.executable, "-m", "pre_commit", "run"]
    cmd += ["--files", *files] if files else ["--all-files"]
    with open(log, "w") as file:
        code = subprocess.run(
            cmd,
            cwd=common.root(),
            env=common.child_env(),
            stdout=file,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code:
        failed = True
        text = log.read_text(errors="replace").splitlines()
        bad = [i for i, line in enumerate(text) if line.rstrip().endswith("Failed")]
        print("\n".join(text[bad[0] :][:40] if bad else text[-20:]))
    verdict = "FAIL" if code else "PASS"
    print(
        f"{verdict} lint, {len(files)} changed files ({time.time() - start:.0f} s). Log: {log}"
    )
    if code:
        print("pre-commit may have reformatted files; review them and run again")

    if not args.no_tests:
        start = time.time()
        code, log = run_pytest(["test"], args.mem, args.timeout)
        if code:
            failed = True
            print(summarize(log))
        verdict = "FAIL" if code else "PASS"
        last = (log.read_text(errors="replace").splitlines() or [""])[-1]
        print(
            f"{verdict} tests: {last.strip('= ')} ({time.time() - start:.0f} s). Log: {log}"
        )
    return FAILED if failed else OK
