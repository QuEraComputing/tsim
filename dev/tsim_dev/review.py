"""`tsim-dev review` and `tsim-dev fuzz`: what a change owes before it merges.

The tier comes from the paths and the size of the diff, not from the author:

- tier 0 (docs, tests, tooling): the `check` gate.
- tier 1 (other changes in `src/tsim`): tier 0 and one review in a fresh context.
- tier 2 (sampling semantics, or a large diff): tier 1, a second review by another
  model, a short differential fuzz run, and a last review round with no findings.

Independently of the tier, a diff on the compile or sampling path (or to the
dependencies) needs a benchmark against the base, and one that can change the active
width needs `bench --width`. A benchmark that finds a regression still counts as run;
the regression has to be explained in the pull request, not removed at any cost.

`review status` reads the evidence that the other commands leave in `.tsim-dev/`
and the pull request's review threads, and prints what is still missing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import common
from .common import FAILED, INFRA, OK, RUNNING, InfraError

# Paths whose changes can alter samples: tier 2.
SAMPLING = (
    "src/tsim/statevector/",
    "src/tsim/noise/",
    "src/tsim/compile/",
    "src/tsim/sampler.py",
    "src/tsim/core/graph.py",
    "src/tsim/utils/kernels.py",
    "src/tsim/utils/rng.py",
)
# Paths whose changes can alter compile or sampling cost: a benchmark is needed.
PERFORMANCE = (
    "src/tsim/circuit.py",
    "src/tsim/core/",
    "src/tsim/compile/",
    "src/tsim/noise/channels.py",
    "src/tsim/statevector/",
    "src/tsim/sampler.py",
    "src/tsim/utils/kernels.py",
    "src/tsim/utils/linalg.py",
    "src/tsim/utils/clifford.py",
    "src/tsim/utils/rng.py",
    "pyproject.toml",
    "uv.lock",
)
# Paths whose changes can alter the active width: `bench --width` is needed.
WIDTH = (
    "src/tsim/circuit.py",
    "src/tsim/core/",
    "src/tsim/compile/",
    "src/tsim/statevector/pbc.py",
    "src/tsim/statevector/plan.py",
    "src/tsim/statevector/program.py",
    "src/tsim/statevector/schedule.py",
)
# Paths a person should read: they change what the gates themselves enforce, or
# what users may rely on.
HUMAN = (
    ".github/workflows/",
    ".claude/settings.json",
    ".claude/hooks/",
    "dev/tsim_dev/review.py",
    "src/tsim/__init__.py",
    "test/unit/test_public_api.py",
)
LARGE = 300  # changed source lines that make a diff tier 2
SPLIT = 600  # changed source lines above which reviews have missed bugs
MAX_ROUNDS = 3


@dataclass
class Plan:
    """The tier of a diff and the reasons for it."""

    tier: int
    base: str
    source_lines: int
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    bench: bool = False  # touches the compile or sampling path
    width: bool = False  # can change the active width


def classify(stat: list[tuple[int, int, str]], base: str = "") -> Plan:
    """Classify a diff given as (added, deleted, path) rows."""
    source = [row for row in stat if row[2].startswith("src/tsim/")]
    source = [row for row in source if not row[2].startswith("src/tsim/external/")]
    lines = sum(added + deleted for added, deleted, _ in source)
    plan = Plan(0, base, lines)
    if source:
        plan.tier = 1
        plan.reasons.append(f"{len(source)} files in src/tsim, {lines} changed lines")
    sampling = sorted({p for _, _, p in source if p.startswith(SAMPLING)})
    if sampling:
        plan.tier = 2
        plan.reasons.append("can change samples: " + ", ".join(sampling[:4]))
    if lines > LARGE:
        plan.tier = 2
        plan.reasons.append(f"more than {LARGE} changed source lines")
    if not source:
        plan.reasons.append("no changes in src/tsim")
    perf = sorted({p for _, _, p in stat if p.startswith(PERFORMANCE)})
    if perf:
        plan.bench = True
        plan.reasons.append("can change performance: " + ", ".join(perf[:4]))
    plan.width = any(p.startswith(WIDTH) for _, _, p in stat)
    if lines > SPLIT:
        plan.notes.append(
            f"SPLIT: {lines} changed source lines. Diffs above {SPLIT} have let "
            "bugs through review here; split it into stacked pull requests."
        )
    human = sorted({p for _, _, p in stat if p.startswith(HUMAN)})
    if human:
        plan.notes.append("HUMAN: a person should read " + ", ".join(human))
    removed = sum(d for _, d, p in stat if p.startswith("test/"))
    added = sum(a for a, _, p in stat if p.startswith("test/"))
    if removed > 20 and removed > added:
        plan.notes.append(
            f"TESTS: {removed} test lines removed, {added} added. Say why in the "
            "pull request; tests are not to be weakened to make a change pass."
        )
    return plan


def pull_request() -> dict | None:
    """Return the open pull request of the current branch, if there is one."""
    done = subprocess.run(
        ["gh", "pr", "view", "--json", "number,baseRefName,headRefOid,state,url"],
        cwd=common.root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        return None
    pr = json.loads(done.stdout)
    return pr if pr.get("state") == "OPEN" else None


def resolve_base(explicit: str | None, pr: dict | None) -> str:
    """Pick the base to diff against: the option, the pull request's base, or main."""
    if explicit:
        return explicit
    if pr:
        return f"origin/{pr['baseRefName']}"
    return "origin/main"


def make_plan(base: str) -> Plan:
    """Classify the diff between the merge base with `base` and HEAD."""
    root = common.root()
    merge_base = common.git("merge-base", "HEAD", base, cwd=root)
    rows = []
    for line in common.git(
        "diff", "--numstat", merge_base, "HEAD", cwd=root
    ).splitlines():
        added, deleted, path = line.split("\t", 2)
        rows.append(
            (
                int(added) if added.isdigit() else 0,
                int(deleted) if deleted.isdigit() else 0,
                path,
            )
        )
    return classify(rows, base)


def head() -> str:
    """Return the full commit id of HEAD."""
    return common.git("rev-parse", "HEAD", cwd=common.root())


def is_dirty() -> bool:
    """Return whether tracked files have uncommitted changes."""
    return bool(
        common.git("status", "--porcelain", "--untracked-files=no", cwd=common.root())
    )


def evidence_path(kind: str) -> Path:
    """Return the file that records the last run of a gate."""
    return common.out_dir("evidence") / f"{kind}.json"


def write_evidence(kind: str, passed: bool, **extra) -> None:
    """Record that a gate ran on the current commit."""
    record = {"head": head(), "dirty": is_dirty(), "passed": passed, **extra}
    evidence_path(kind).write_text(json.dumps(record))


def evidence_record(kind: str) -> dict:
    """Return the last recorded run of a gate, or an empty dict."""
    path = evidence_path(kind)
    return json.loads(path.read_text()) if path.exists() else {}


def regressions(record: dict) -> str:
    """Describe the regressions a benchmark run found, or return ""."""
    found = [
        f"{record[key]} {label}"
        for key, label in (
            ("slower", "slower"),
            ("wider", "wider"),
            ("kept_differs", "with a different kept fraction"),
        )
        if record.get(key)
    ]
    return ", ".join(found)


def read_evidence(kind: str) -> tuple[bool, str]:
    """Return whether a gate passed on exactly this commit, and a description."""
    path = evidence_path(kind)
    if not path.exists():
        return False, "not run"
    record = json.loads(path.read_text())
    if record["head"] != head() or record["dirty"] or is_dirty():
        return False, "ran on other code than the current commit; run it again"
    return bool(record["passed"]), "passed" if record["passed"] else "failed"


def reviews_path() -> Path:
    """Return the review log of the current branch."""
    branch = common.git("rev-parse", "--abbrev-ref", "HEAD", cwd=common.root())
    return common.out_dir("reviews") / (branch.replace("/", "__") + ".jsonl")


def read_reviews() -> list[dict]:
    """Return the recorded review rounds of the current branch."""
    path = reviews_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def family(model: str) -> str:
    """Reduce a model name to its family, so two Claude models count as one."""
    name = model.lower()
    for key, value in (
        ("claude", "anthropic"),
        ("opus", "anthropic"),
        ("sonnet", "anthropic"),
        ("fable", "anthropic"),
        ("haiku", "anthropic"),
        ("gpt", "openai"),
        ("codex", "openai"),
        ("astra", "openai"),
        ("gemini", "google"),
    ):
        if key in name:
            return value
    return name


def review_gaps(tier: int, reviews: list[dict]) -> list[str]:
    """Return what the recorded reviews still lack for this tier."""
    if tier == 0:
        return []
    if not reviews:
        return ["no review recorded; see the reviewing-tsim-changes skill"]
    gaps = []
    if tier == 2:
        if len({family(r["model"]) for r in reviews}) < 2:
            gaps.append("tier 2 needs a second review by a model from another family")
        last = reviews[-1]
        if last["findings"]:
            if len(reviews) >= MAX_ROUNDS:
                gaps.append(
                    f"round {len(reviews)} still had findings. Stop reviewing: "
                    "rethink the approach or split the change, and tell the user"
                )
            else:
                gaps.append(
                    "the last round had findings; after the fixes, run a round on "
                    "the new diff only, until one has none"
                )
    return gaps


def open_threads(pr: dict) -> tuple[list[str], list[str]]:
    """Return unanswered review threads and reviewers that have not reported yet.

    A thread counts as answered when it is resolved or has a reply.
    """
    query = """query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){
      pullRequest(number:$number){
        reviewRequests(first:20){nodes{requestedReviewer{__typename ... on Bot{login} ... on User{login}}}}
        reviewThreads(first:100){nodes{isResolved path line
          comments(first:30){nodes{author{login} body url}}}}}}}"""
    owner, name = (
        common.git("remote", "get-url", "origin", cwd=common.root())
        .removesuffix(".git")
        .replace(":", "/")
        .split("/")[-2:]
    )
    done = subprocess.run(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"number={pr['number']}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        raise InfraError(f"gh api graphql: {done.stderr.strip()[-200:]}")
    data = json.loads(done.stdout)["data"]["repository"]["pullRequest"]
    threads = []
    for thread in data["reviewThreads"]["nodes"]:
        comments = thread["comments"]["nodes"]
        if thread["isResolved"] or len(comments) != 1:
            continue
        first = comments[0]
        text = " ".join(first["body"].split())
        text = text.split("</sub></sub>")[-1].strip(" *")[:90]
        author = (first["author"] or {}).get("login", "?")
        threads.append(
            f"{thread['path']}:{thread['line']} [{author}] {text}\n      {first['url']}"
        )
    pending = [
        (node["requestedReviewer"] or {}).get("login", "?")
        for node in data["reviewRequests"]["nodes"]
        if (node["requestedReviewer"] or {}).get("__typename") == "Bot"
    ]
    return threads, pending


def ci_state(pr: dict) -> tuple[str, list[str]]:
    """Return "pass", "fail" or "pending" for the pull request's checks."""
    done = subprocess.run(
        ["gh", "pr", "checks", str(pr["number"]), "--json", "name,bucket"],
        cwd=common.root(),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        checks = json.loads(done.stdout)
    except ValueError:
        return "pending", ["no checks reported yet"]
    failed = [c["name"] for c in checks if c["bucket"] in ("fail", "cancel")]
    if failed:
        return "fail", failed
    waiting = [c["name"] for c in checks if c["bucket"] == "pending"]
    return ("pending", waiting) if waiting else ("pass", [])


def print_plan(plan: Plan) -> None:
    """Print the tier, why, and what it requires."""
    print(f"tier {plan.tier} against {plan.base}: " + "; ".join(plan.reasons))
    for note in plan.notes:
        print(note)
    steps = ["uv run tsim-dev check"]
    if plan.bench:
        steps.append(
            f"uv run tsim-dev bench --ref {plan.base}; explain any regression in the PR"
        )
    if plan.width:
        steps.append(f"uv run tsim-dev bench --width --ref {plan.base}")
    if plan.tier >= 1:
        steps.append("one review in a fresh context (reviewing-tsim-changes skill)")
    if plan.tier >= 2:
        steps.append("a second review by a model from another family")
        steps.append("uv run tsim-dev fuzz")
        steps.append(
            f"a last review round with no findings (at most {MAX_ROUNDS} rounds)"
        )
    steps.append("every review thread on the pull request answered")
    print("required:")
    for step in steps:
        print(f"  - {step}")


def plan(args) -> int:
    """Print the tier of the current branch and what it owes."""
    try:
        print_plan(make_plan(resolve_base(args.base, pull_request())))
    except InfraError as error:
        common.die(str(error))
    return OK


def record(args) -> int:
    """Record one finished review round for the current branch."""
    reviews = read_reviews()
    entry = {
        "round": len(reviews) + 1,
        "model": args.model,
        "findings": args.findings,
        "head": head(),
        "note": args.note,
    }
    with open(reviews_path(), "a") as file:
        file.write(json.dumps(entry) + "\n")
    print(
        f"recorded round {entry['round']}: {args.model}, {args.findings} confirmed findings"
    )
    return OK


def status(args) -> int:
    """Check every requirement of the tier and print what is missing."""
    try:
        pr = pull_request()
        current = make_plan(resolve_base(args.base, pr))
        print(
            f"tier {current.tier} against {current.base}: " + "; ".join(current.reasons)
        )
        missing: list[str] = []
        waiting: list[str] = []

        def line(ok: bool, text: str, pending: bool = False) -> None:
            mark = "ok  " if ok else ("wait" if pending else "TODO")
            print(f"  [{mark}] {text}")
            if not ok:
                (waiting if pending else missing).append(text)

        for note in current.notes:
            print(f"  [note] {note}")
        checked, detail = read_evidence("check")
        ci, names = ci_state(pr) if pr else ("none", [])
        if ci == "pass" and not checked:
            line(True, "check: CI passed on the pull request")
        else:
            line(checked, f"check: {detail} (uv run tsim-dev check)")
        if pr and ci != "pass":
            line(False, f"CI {ci}: {', '.join(names[:4])}", pending=ci == "pending")
        for kind, needed, command in (
            ("bench", current.bench, f"uv run tsim-dev bench --ref {current.base}"),
            (
                "width",
                current.width,
                f"uv run tsim-dev bench --width --ref {current.base}",
            ),
        ):
            if not needed:
                continue
            ran, detail = read_evidence(kind)
            line(ran, f"{kind}: {detail} ({command})")
            found = regressions(evidence_record(kind)) if ran else ""
            if found:
                print(
                    f"  [note] {kind} found regressions ({found}): confirm, profile, "
                    "and explain them in the pull request"
                )
        if current.tier >= 2:
            fuzzed, detail = read_evidence("fuzz")
            line(fuzzed, f"fuzz: {detail} (uv run tsim-dev fuzz)")
        reviews = read_reviews()
        summary = ", ".join(f"{r['model']}: {r['findings']}" for r in reviews) or "none"
        gaps = review_gaps(current.tier, reviews)
        line(
            not gaps, f"reviews ({summary})" + "".join(f"\n         {g}" for g in gaps)
        )
        if pr:
            threads, pending = open_threads(pr)
            line(
                not threads, f"{len(threads)} unanswered review threads on {pr['url']}"
            )
            for thread in threads[:12]:
                print(f"    - {thread}")
            if pending:
                line(False, "waiting for " + ", ".join(pending), pending=True)
        else:
            print("  [note] no open pull request: review threads and CI not checked")
    except InfraError as error:
        common.die(str(error))
    if missing:
        print(f"NOT READY: {len(missing)} items to do")
        return FAILED
    if waiting:
        print("WAIT: nothing to do yet, but checks or reviewers are still running")
        return RUNNING
    print("READY: every requirement of this tier is met")
    return OK


def fuzz(args) -> int:
    """Run the differential fuzzer for a fixed time and record the result."""
    out = common.out_dir("fuzz")
    worker = Path(__file__).resolve().parent / "_fuzz.py"
    cmd = [sys.executable, str(worker), "--seconds", str(args.seconds)]
    cmd += ["--start", str(args.start), "--out", str(out)]
    env = common.child_env(1)
    env["PYTHONPATH"] = str(common.root())
    done = subprocess.run(
        common.capped(cmd, mem=args.mem, timeout=int(args.seconds) + 300),
        cwd=common.root(),
        env=env,
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
    if result is None:
        reason = common.explain_exit(done.returncode, args.mem, None)
        print(f"ERROR: fuzz did not finish ({reason}): {done.stderr.strip()[-300:]}")
        return INFRA
    failures = result["failures"]
    for failure in failures[:10]:
        print(
            f"FAILURE seed={failure['seed']}: {failure['problem']}\n  {failure['circuit']}"
        )
    write_evidence("fuzz", not failures, circuits=result["circuits"])
    verdict = "FAIL" if failures else "PASS"
    print(
        f"{verdict} fuzz: {result['circuits']} circuits from seed {result['start']} in "
        f"{args.seconds:g} s, {len(failures)} failures"
    )
    return FAILED if failures else OK
