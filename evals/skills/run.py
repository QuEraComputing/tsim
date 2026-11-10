"""Run triggering and behavioral evals of the Tsim skill with the Claude Code CLI.

    python evals/skills/run.py triggers
    python evals/skills/run.py cases [--case NAME] [--runs N]
    python evals/skills/run.py grade RESULTS_DIR

See README.md in this directory. Every agent run gets a fresh working directory under the
results directory, which is outside the repository so that the agent does not read Tsim's own
AGENTS.md or CLAUDE.md; the skills are copied into it only for the ``with`` arm. Runs are
capped with ``systemd-run`` (memory, CPU) and ``timeout``, because agents run real
simulations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SKILLS = REPO / "skills"
SKILL_NAMES = sorted(p.parent.name for p in SKILLS.glob("*/SKILL.md"))
CIRCUITS = HERE / "circuits"
# NOTE: Claude Code reads CLAUDE.md and AGENTS.md from every parent directory, so runs
# inside this repository would see Tsim's developer notes even without the skill.
DEFAULT_RESULTS = Path.home() / ".cache" / "tsim-skill-evals"

PREAMBLE = """
Work in the current directory; input files are under inputs/. Python with Tsim, Stim,
PyMatching, Tesseract and sinter is available as `python`. Use at most {cpus} CPU cores and
{mem} of memory. Save the scripts and data files you create in the current directory.
"""


# ----------------------------------------------------------------------------- agents


def agent_command(prompt: str, args, *, tools: str) -> list[str]:
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose"]
    cmd += ["--max-turns", str(args.max_turns), "--permission-mode", "acceptEdits"]
    cmd += ["--allowedTools", *tools.split()]
    if args.model:
        cmd += ["--model", args.model]
    return cmd


def capped(cmd: list[str], args) -> list[str]:
    if args.no_cap:
        return ["timeout", str(args.timeout), *cmd]
    if shutil.which("systemd-run") is None:
        sys.exit(
            "systemd-run not found; pass --no-cap to run without memory and CPU limits"
        )
    return [
        "systemd-run", "--user", "--scope", "-q",
        "-p", f"MemoryMax={args.mem}", "-p", "MemorySwapMax=0",
        "-p", f"CPUQuota={100 * args.cpus}%",
        "timeout", str(args.timeout), *cmd,
    ]  # fmt: skip


def prepare_workdir(work: Path, *, with_skills: bool, inputs: dict, args) -> dict:
    work.mkdir(parents=True)
    if inputs:
        (work / "inputs").mkdir()
    for name, spec in inputs.items():
        text = (CIRCUITS / spec["circuit"]).read_text()
        if spec.get("s_to_t"):
            text = re.sub(r"^(\s*)S(_DAG)?(?=\s)", r"\1T\2", text, flags=re.MULTILINE)
        (work / "inputs" / name).write_text(text)
    if with_skills:
        shutil.copytree(SKILLS, work / ".claude" / "skills")
    bin_dir = work / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "python"
    shim.write_text(
        '#!/bin/sh\nexec uv run --quiet --project "$TSIM_EVAL_PROJECT" python "$@"\n'
    )
    shim.chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)
    env = dict(os.environ)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        TSIM_EVAL_PROJECT=str(REPO),
        NUMBA_NUM_THREADS=str(args.cpus),
        OMP_NUM_THREADS=str(args.cpus),
        JAX_PLATFORMS="cpu",
    )
    return env


def run_agent(
    prompt: str, out: Path, *, with_skills: bool, inputs, args, tools
) -> dict:
    work = out / "work"
    env = prepare_workdir(work, with_skills=with_skills, inputs=inputs, args=args)
    cmd = capped(agent_command(prompt, args, tools=tools), args)
    start = time.monotonic()
    with (
        open(out / "transcript.jsonl", "w") as stdout,
        open(out / "stderr.log", "w") as stderr,
    ):
        proc = subprocess.run(
            cmd,
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
        )
    meta = {
        "with_skills": with_skills,
        "exit_code": proc.returncode,
        "seconds": round(time.monotonic() - start, 1),
    }
    meta.update(parse_transcript(out))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ----------------------------------------------------------------------------- transcripts


def _events(path: Path):
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_transcript(out: Path) -> dict:
    """Return the final answer, the code the agent ran or wrote, skills it loaded, and cost."""
    code: list[str] = []
    skills: set[str] = set()
    answer, cost, ran = "", None, False

    def note_path(text: str) -> None:
        for name in SKILL_NAMES:
            if f"skills/{name}/" in text:
                skills.add(name)

    for ev in _events(out / "transcript.jsonl"):
        if ev.get("type") == "assistant":
            for item in ev["message"].get("content", []):
                if item.get("type") != "tool_use":
                    continue
                inp = item.get("input", {})
                if item["name"] == "Skill":
                    skills.add(str(inp.get("skill", "")).split(":")[-1])
                note_path(json.dumps(inp))
                code += [str(v) for v in inp.values() if isinstance(v, str)]
        elif ev.get("type") == "result":
            answer, cost = ev.get("result") or "", ev.get("total_cost_usd")
            ran = True  # also after the turn limit, which is not a failure to run
    for script in sorted((out / "work").rglob("*.py")):
        if ".claude" not in script.parts:
            code.append(script.read_text(errors="ignore"))
    (out / "code.txt").write_text("\n\n".join(code))
    (out / "answer.txt").write_text(answer)
    # NOTE: Claude can also invoke unrelated skills installed for the user.
    return {
        "skills_loaded": sorted(skills & set(SKILL_NAMES)),
        "cost_usd": cost,
        "ran": ran,
    }


# ----------------------------------------------------------------------------- triggers


def cmd_triggers(args) -> None:
    queries = json.loads((HERE / "triggers.json").read_text())
    if args.only:
        queries = [queries[i] for i in args.only]
    root = results_dir(args, "triggers")
    jobs = [(i, q, r) for i, q in enumerate(queries) for r in range(args.runs)]

    def one(job):
        i, q, r = job
        out = root / f"q{i:02d}-run{r + 1}"
        out.mkdir(parents=True)
        meta = run_agent(
            q["query"], out, with_skills=True, inputs={}, args=args,
            tools="Skill Read Glob Grep",
        )  # fmt: skip
        loaded = set(meta["skills_loaded"])
        # NOTE: a run that never started loads nothing, which must not pass a negative query.
        ok = bool(loaded) if q["load"] else meta["ran"] and not loaded
        return {
            "query": q["query"],
            "expected": q["load"],
            "loaded": sorted(loaded),
            "ok": ok,
        }

    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(one, jobs))
    (root / "triggers.json").write_text(json.dumps(rows, indent=2))
    lines = ["| ok | should load | loaded | query |", "|---|---|---|---|"]
    for row in rows:
        mark = "✓" if row["ok"] else "✗"
        loaded = ", ".join(row["loaded"]) or "none"
        lines.append(
            f"| {mark} | {'yes' if row['expected'] else 'no'} | {loaded} | "
            f"{row['query'][:70]} |"
        )
    score = sum(r["ok"] for r in rows) / len(rows)
    summary = f"# Triggering\n\n{score:.0%} correct\n\n" + "\n".join(lines)
    (root / "summary.md").write_text(summary + "\n")
    print(summary)
    print(f"\nResults: {root}")


# ----------------------------------------------------------------------------- cases


def load_case(case_dir: Path) -> tuple[dict, object]:
    case = tomllib.loads((case_dir / "case.toml").read_text())
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(
        f"check_{case_dir.name}", case_dir / "check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return case, module


def grade(run_dir: Path, checker) -> dict:
    answer = (run_dir / "answer.txt").read_text()
    code = (run_dir / "code.txt").read_text()
    results = checker.check(answer, code, run_dir / "work")
    grades = {
        name: {"passed": bool(ok), "evidence": ev} for name, (ok, ev) in results.items()
    }
    (run_dir / "grades.json").write_text(json.dumps(grades, indent=2))
    return grades


def cmd_cases(args) -> None:
    case_dirs = sorted(
        p for p in (HERE / "cases").iterdir() if (p / "case.toml").exists()
    )
    if args.case:
        case_dirs = [p for p in case_dirs if p.name in args.case]
    root = results_dir(args, "cases")
    preamble = PREAMBLE.format(cpus=args.cpus, mem=args.mem)
    jobs = [
        (case_dir, arm, r)
        for case_dir in case_dirs
        for arm in args.arms.split(",")
        for r in range(args.runs)
    ]

    def one(job):
        case_dir, arm, r = job
        case, _ = load_case(case_dir)
        out = root / case_dir.name / f"{arm}-run{r + 1}"
        out.mkdir(parents=True)
        run_agent(
            case["prompt"].strip() + "\n" + preamble, out,
            with_skills=(arm == "with"), inputs=case.get("inputs", {}), args=args,
            tools="Bash Read Write Edit Glob Grep Skill",
        )  # fmt: skip

    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(one, jobs))
    summarize(root)


def summarize(root: Path) -> None:
    lines = [
        "| case | arm | runs | checks passed | skill loaded | cost (USD) | minutes |",
        "|---|---|---|---|---|---|---|",
    ]
    failures = []
    for case_root in sorted(p for p in root.iterdir() if p.is_dir()):
        _, checker = load_case(HERE / "cases" / case_root.name)
        for arm in ("with", "without"):
            runs = sorted(case_root.glob(f"{arm}-run*"))
            if not runs:
                continue
            passed = total = loaded = 0
            cost = minutes = 0.0
            for run in runs:
                grades = grade(run, checker)
                meta = json.loads((run / "meta.json").read_text())
                passed += sum(g["passed"] for g in grades.values())
                total += len(grades)
                loaded += bool(meta["skills_loaded"])
                cost += meta["cost_usd"] or 0.0
                minutes += meta["seconds"] / 60
                failures += [
                    f"- {case_root.name} {run.name}: {name}: {g['evidence']}"
                    for name, g in grades.items()
                    if not g["passed"]
                ]
            n = len(runs)
            lines.append(
                f"| {case_root.name} | {arm} | {n} | {passed}/{total} | {loaded}/{n} | "
                f"{cost / n:.2f} | {minutes / n:.1f} |"
            )
    summary = "# Behavioral evals\n\n" + "\n".join(lines) + "\n\n## Failed checks\n\n"
    summary += "\n".join(failures) or "none"
    (root / "summary.md").write_text(summary + "\n")
    print(summary)
    print(f"\nResults: {root}")


def cmd_grade(args) -> None:
    summarize(Path(args.results).resolve())


# ----------------------------------------------------------------------------- CLI


def results_dir(args, kind: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.results).expanduser().resolve() / f"{stamp}-{kind}"
    root.mkdir(parents=True)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("triggers", "cases"):
        p = sub.add_parser(name)
        p.add_argument("--model", help="model passed to claude")
        p.add_argument(
            "--results",
            default=str(DEFAULT_RESULTS),
            help="results directory, outside the repository (default %(default)s)",
        )
        p.add_argument(
            "--runs", type=int, default=1, help="runs per query or case and arm"
        )
        p.add_argument("-j", "--jobs", type=int, default=1, help="agent runs at once")
        p.add_argument(
            "--mem", default="8G", help="memory cap per run (systemd MemoryMax)"
        )
        p.add_argument("--cpus", type=int, default=4, help="CPU cap per run")
        p.add_argument("--timeout", type=int, default=1800, help="seconds per run")
        p.add_argument("--max-turns", type=int, default=80, help="turn limit")
        p.add_argument(
            "--no-cap", action="store_true", help="run without systemd limits"
        )
    triggers = sub.choices["triggers"]
    triggers.add_argument(
        "--only", type=int, action="append", help="query index (repeatable)"
    )
    triggers.set_defaults(func=cmd_triggers, max_turns=4, timeout=300)
    cases = sub.choices["cases"]
    cases.add_argument("--case", action="append", help="case name (repeatable)")
    cases.add_argument(
        "--arms",
        default="with,without",
        choices=["with", "without", "with,without"],
        help="arms to run (default both: with,without)",
    )
    cases.set_defaults(func=cmd_cases)
    g = sub.add_parser("grade", help="regrade an existing cases results directory")
    g.add_argument("results")
    g.set_defaults(func=cmd_grade)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
