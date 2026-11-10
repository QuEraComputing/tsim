"""Tests for the development CLI in `dev/` and the agent hook in `.claude/`."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tsim_dev import bench, checks, cli, common, corpus, review, width

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / ".claude" / "hooks" / "guard_bash.py"


FAKE_BENCH = """
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Case:
    id: str
    path: Path
    sampler: str
    postselect: str
    num_detectors: int = 2
    arxiv: str = "0000.00000"
    widths: dict = field(default_factory=dict, compare=False)

    def width(self, backend="tsim"):
        return self.widths.get(backend)


def load(root=ROOT):
    stim = root / "papers" / "p" / "stim"
    return {
        "p/narrow": Case("p/narrow", stim / "narrow.stim", "detector", "all", widths={"tsim": 4, "clifft": 9}),
        "p/wide": Case("p/wide", stim / "wide.stim", "detector", "none", widths={"tsim": 26}),
    }


def mask_indices(case):
    return list(range(case.num_detectors)) if case.postselect == "all" else []


def cases(suite=None, *, max_width=None, width_of="tsim", only=None, root=ROOT):
    selected = list(load(root).values())
    if suite not in (None, "corpus"):
        if suite != "quick":
            raise KeyError(f"unknown suite {suite!r}")
        selected = selected[:1]
    if max_width is not None:
        selected = [c for c in selected if (c.width(width_of) or 10**9) <= max_width]
    return selected
"""


@pytest.fixture
def fake_bench(tmp_path, monkeypatch):
    package = tmp_path / "tsim_bench"
    package.mkdir()
    (package / "__init__.py").write_text(FAKE_BENCH)
    monkeypatch.setenv("TSIM_BENCH_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "tsim_bench", raising=False)
    for cached in (corpus.root, corpus.package):
        cached.cache_clear()
    yield tmp_path
    for cached in (corpus.root, corpus.package):
        cached.cache_clear()


def _bench_args(*argv):
    return cli.build_parser().parse_args(["bench", *argv])


def test_corpus_circuits_are_postselected_unless_turned_off(fake_bench):
    (case,) = bench.select(_bench_args())  # the quick suite
    assert (case.id, case.postselect, case.width) == ("p/narrow", "recorded", 4)
    command = bench.worker_command(Path("python"), case, _bench_args(), "bench")
    assert command[command.index("--case") + 1] == "p/narrow"
    assert "--no-postselect" not in command

    (case,) = bench.select(_bench_args("--no-postselect"))
    assert case.postselect == "none"
    command = bench.worker_command(Path("python"), case, _bench_args(), "bench")
    assert "--no-postselect" in command


def test_max_width_uses_the_recorded_width_of_one_backend(fake_bench):
    everything = bench.select(_bench_args("--suite", "corpus"))
    assert [case.id for case in everything] == ["p/narrow", "p/wide"]
    narrow = bench.select(_bench_args("--suite", "corpus", "--max-width", "8"))
    assert [case.id for case in narrow] == ["p/narrow"]
    by_clifft = ["--suite", "corpus", "--max-width", "8", "--width-of", "clifft"]
    with pytest.raises(common.InfraError, match="no circuits"):
        bench.select(_bench_args(*by_clifft))


def test_circuits_are_named_by_id_file_stem_or_path(fake_bench, tmp_path):
    assert [c.id for c in bench.select(_bench_args("wide"))] == ["p/wide"]
    assert [c.id for c in bench.select(_bench_args("p/narrow"))] == ["p/narrow"]
    stim = tmp_path / "mine.stim"
    stim.write_text("H 0\nM 0\n")
    (case,) = bench.select(_bench_args(str(stim), "--postselect", "all"))
    assert (case.id, case.postselect) == (None, "all")
    command = bench.worker_command(Path("python"), case, _bench_args(), "bench")
    assert command[command.index("--circuit") + 1] == str(stim)
    with pytest.raises(common.InfraError, match="unknown circuit"):
        bench.select(_bench_args("nope"))
    with pytest.raises(common.InfraError, match="unknown suite"):
        bench.select(_bench_args("--suite", "nope"))


@pytest.mark.parametrize(
    ("new", "ref", "verdict"),
    [
        ([100, 101, 99], [100, 100, 100], "same"),
        ([120, 121, 119], [100, 100, 100], "faster"),
        ([80, 81, 79], [100, 100, 100], "slower"),
        ([130, 130, 95], [100, 100, 100], "noisy"),
        ([70, 70, 105], [100, 100, 100], "noisy"),
        ([104, 104, 104], [100, 100, 100], "same"),
    ],
)
def test_compare_needs_every_round_to_agree(new, ref, verdict):
    result = bench.compare(new, ref, tolerance=0.05)
    assert result["verdict"] == verdict
    assert result["low"] <= result["ratio"] <= result["high"]


def test_changed_source_files_map_to_their_tests():
    assert checks.tests_for(["src/tsim/sampler.py"]) == [
        "test/integration/test_sampler.py",
        "test/unit/test_sampler.py",
    ]
    assert checks.tests_for(["src/tsim/statevector/plan.py"]) == [
        "test/unit/statevector/test_plan.py"
    ]
    assert checks.tests_for(["test/unit/test_sampler.py", "README.md"]) == [
        "test/unit/test_sampler.py"
    ]
    assert checks.tests_for(["dev/tsim_dev/bench.py"]) == ["test/unit/test_dev_cli.py"]
    assert checks.tests_for(["README.md"]) == []


def test_changes_without_a_matching_test_run_the_full_suite():
    assert checks.tests_for(["src/tsim/__init__.py"]) is None
    assert checks.tests_for(["test/helpers/statevector.py"]) is None


def test_counts_are_formatted_the_short_way():
    assert [common.si(x) for x in (950, 3200, 2_100_000, 28.4e6)] == [
        "950",
        "3.2k",
        "2.1M",
        "28.4M",
    ]


def test_everything_after_the_double_dash_is_the_command(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.jobs, "run", lambda args: seen.update(vars(args)) or 0)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--mem", "2G", "--", "python", "--mem", "x.py"])
    assert exit_info.value.code == 0
    assert seen["mem"] == "2G"
    assert seen["command"] == ["python", "--mem", "x.py"]


def test_pytest_summary_keeps_the_first_failure_and_the_last_line(tmp_path):
    log = tmp_path / "test.log"
    log.write_text(
        "..F.\n"
        "=== FAILURES ===\n"
        "____ test_x ____\n"
        "E   assert 1 == 2\n"
        "=== short test summary info ===\n"
        "FAILED test/unit/test_x.py::test_x - assert 1 == 2\n"
        "1 failed, 3 passed in 0.10s\n"
    )
    summary = checks.summarize(log)
    assert "E   assert 1 == 2" in summary
    assert summary.splitlines()[-1] == "1 failed, 3 passed in 0.10s"


@pytest.mark.parametrize(
    ("tool_input", "blocked"),
    [
        ({"command": "pkill -f fuzz.py"}, True),
        ({"command": "pkill -9 -f fuzz"}, True),
        ({"command": "nohup python x.py > log &"}, True),
        ({"command": "uv run pytest test", "run_in_background": True}, True),
        ({"command": "uv run tsim-dev bench", "run_in_background": True}, False),
        ({"command": "systemd-run --user --scope python x.py &"}, False),
        ({"command": "uv run pytest test/unit -q"}, False),
        ({"command": "python a.py && python b.py"}, False),
        ({"command": "kill 1234"}, False),
        ({"command": "sleep 5 &"}, False),
    ],
)
def test_the_bash_hook_blocks_self_kills_and_uncapped_background_compute(
    tool_input, blocked
):
    done = subprocess.run(  # noqa: S603
        [sys.executable, str(HOOK)],
        input=json.dumps({"tool_name": "Bash", "tool_input": tool_input}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == (2 if blocked else 0)
    assert bool(done.stderr) == blocked


def test_dev_skills_have_valid_frontmatter():
    skills = sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md"))
    assert len(skills) >= 3
    for path in skills:
        text = path.read_text()
        assert text.startswith("---\n")
        front = yaml.safe_load(text.split("---\n")[1])
        assert front["name"] == path.parent.name
        # NOTE: hides development skills from `npx skills add QuEraComputing/tsim`.
        assert front["metadata"]["internal"] is True
        assert 0 < len(front["description"]) <= 1024
        assert len(text.splitlines()) < 500


def test_sides_carry_their_own_settings(fake_bench):
    side = bench.parse_side(".:threads=10,strategy=cat5")
    assert (side.python, side.threads, side.strategy) == (
        Path(sys.executable),
        10,
        "cat5",
    )
    assert side.describe().endswith("threads=10 strategy=cat5")
    assert bench.side_env(side, 10)["NUMBA_NUM_THREADS"] == "10"
    # A python executable is used as given: resolving a venv's symlink would lose the venv.
    python = Path(sys.executable).absolute()
    assert bench.parse_side(f"{python}:threads=2").python == python
    for bad in (".:cores=2", ".:threads=0", ".:device=gpu", ".:threads"):
        with pytest.raises(common.InfraError):
            bench.parse_side(bad)


def test_profile_and_inspect_accept_a_corpus_dir():
    for command in ("profile", "inspect"):
        args = cli.build_parser().parse_args([command, "qv_q10"])
        assert args.corpus_dir is None


def test_paths_follow_the_manifest_and_no_postselect(fake_bench, tmp_path):
    stim = tmp_path / "papers" / "p" / "stim" / "narrow.stim"
    stim.parent.mkdir(parents=True)
    stim.write_text("H 0\nM 0\n")
    (case,) = bench.select(_bench_args(str(stim)))
    assert (case.id, case.postselect) == ("p/narrow", "recorded")
    other = tmp_path / "mine.stim"
    other.write_text("H 0\nM 0\n")
    args = _bench_args(str(other), "--postselect", "all", "--no-postselect")
    assert bench.select(args)[0].postselect == "none"


def test_cases_carry_the_recorded_tsim_width(fake_bench):
    (case,) = bench.select(_bench_args("narrow", "--width-of", "clifft"))
    assert (case.width, case.tsim_width) == (9, 4)


def test_sides_with_different_kept_fractions_are_flagged():
    def rounds(p, shots=100_000):
        return [{"acceptance": p, "shots": shots}] * 3

    assert bench.same_acceptance(rounds(0.144), rounds(0.1445))
    assert not bench.same_acceptance(rounds(0.144), rounds(0.16))
    assert bench.same_acceptance([{"acceptance": None}], rounds(0.5))


def test_the_corpus_export_is_complete_and_shared(tmp_path, monkeypatch):
    import threading

    source = tmp_path / "source"
    (source / "tsim_bench").mkdir(parents=True)
    (source / "tsim_bench" / "__init__.py").write_text("")
    common.git("init", "-q", cwd=source)
    common.git("add", ".", cwd=source)
    common.git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "-m",
        "x",
        cwd=source,
    )
    rev = common.git("rev-parse", "HEAD", cwd=source)
    monkeypatch.setattr(corpus, "URL", str(source))
    monkeypatch.setenv("TSIM_DEV_CACHE", str(tmp_path / "cache"))
    results, errors = [], []

    def export():
        try:
            results.append(corpus._export(rev))
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=export) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(set(results)) == 1
    assert (results[0] / corpus.COMPLETE).exists()
    assert (results[0] / "tsim_bench" / "__init__.py").exists()
    with pytest.raises(common.InfraError, match="has no commit"):
        corpus._export("0" * 40)


@pytest.mark.parametrize(
    ("new", "ref", "expected"),
    [
        ({"width": 4}, {"width": 4}, "same"),
        ({"width": 3}, {"width": 4}, "narrower"),
        ({"width": 5}, {"width": 4}, "wider"),
        ({"width": 40, "refused": True}, {"width": 26}, "wider"),
        ({"width": 26}, {"width": 40, "refused": True}, "narrower"),
        ({"width": 63, "refused": True}, {"width": 63, "refused": True}, "same"),
        ({"width": 20, "refused": True}, {"width": 26}, "unknown"),
        ({"error": "x"}, {"width": 4}, "error"),
    ],
)
def test_width_verdicts_treat_refused_widths_as_lower_bounds(new, ref, expected):
    assert width.verdict(new, ref) == expected


def test_width_mode_resumes_after_the_worker_dies(fake_bench, monkeypatch):
    calls = []

    def fake_stream(cmd, env, case_timeout, outcome):
        cases = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--case"]
        calls.append(cases)
        for case in cases:
            yield f"START {case}"
            if case == "p/narrow" and len(calls) == 1:
                outcome.timed_out = True
                return
            yield "RESULT " + json.dumps({"name": case, "width": 3})

    monkeypatch.setattr(width, "_stream", fake_stream)
    cases = bench.select(_bench_args("--width", "--suite", "corpus"))
    results = width.measure(bench.parse_side("."), cases, _bench_args("--width"))
    assert results["p/narrow"]["error"].startswith("compile took over")
    assert results["p/wide"] == {"width": 3}
    assert calls == [["p/narrow", "p/wide"], ["p/wide"]]


def test_the_width_stream_stops_a_stuck_compile():
    script = "import time; print('START a', flush=True); time.sleep(30)"
    outcome = width.Exit()
    lines = list(width._stream([sys.executable, "-c", script], None, 1, outcome))
    assert lines == ["START a"] and outcome.timed_out


def test_shot_benchmarks_compare_and_check_the_active_width(fake_bench, monkeypatch):
    widths = {"new": [6, 6, 6], "ref": [6, 6, 6]}

    def fake_measure(side, case, args, mode="bench"):
        result = {"shots_per_s": 1e6, "compile_s": 0.1, "acceptance": None, "shots": 1}
        result["active_width"] = widths[side.label].pop(0)
        return result

    monkeypatch.setattr(bench, "measure", fake_measure)
    (case,) = bench.select(_bench_args())
    sides = [bench.Side("new", Path("python")), bench.Side("ref", Path("python"))]
    args = _bench_args("--rounds", "2")
    args.rounds = 2
    record = bench.run_case(case, sides, args)
    assert (record["k"], record["ref_k"], record["width"]) == (6, 6, "same")

    widths.update(new=[7, 7, 7], ref=[6, 6, 6])
    record = bench.run_case(case, sides, args)
    assert record["width"] == "wider"
    assert "WIDER: k 6 -> 7" in bench.format_row(record, has_ref=True)

    widths.update(new=[6, 6, 7], ref=[6, 6, 6])
    assert "changed between runs" in bench.run_case(case, sides, args)["error"]


@pytest.mark.parametrize(
    ("stat", "tier"),
    [
        ([(10, 2, "docs/index.mdx"), (40, 0, "test/unit/test_x.py")], 0),
        ([(200, 0, "dev/tsim_dev/bench.py")], 0),
        ([(30, 5, "src/tsim/circuit.py")], 1),
        ([(3, 1, "src/tsim/statevector/plan.py")], 2),
        ([(2, 2, "src/tsim/sampler.py")], 2),
        ([(250, 60, "src/tsim/circuit.py")], 2),
        ([(900, 0, "src/tsim/external/vec_sim/vec_sampler.py")], 0),
    ],
)
def test_the_review_tier_follows_paths_and_size(stat, tier):
    assert review.classify(stat).tier == tier


def test_review_plans_flag_large_diffs_gate_files_and_removed_tests():
    plan = review.classify(
        [
            (700, 0, "src/tsim/circuit.py"),
            (1, 1, ".github/workflows/ci.yml"),
            (2, 80, "test/unit/test_circuit.py"),
        ]
    )
    assert [note.split(":")[0] for note in plan.notes] == ["SPLIT", "HUMAN", "TESTS"]
    assert review.classify([(5, 5, "src/tsim/circuit.py")]).notes == []


def _rounds(*rounds):
    return [{"model": model, "findings": findings} for model, findings in rounds]


def test_review_requirements_grow_with_the_tier():
    assert review.review_gaps(0, []) == []
    assert review.review_gaps(1, []) != []
    assert review.review_gaps(1, _rounds(("opus-5.5", 3))) == []
    # Two Claude models are one family; tier 2 needs another one.
    assert review.review_gaps(2, _rounds(("opus-5.5", 0), ("claude-fable", 0))) != []
    assert review.review_gaps(2, _rounds(("opus-5.5", 2), ("gpt-6-astra", 0))) == []


@pytest.mark.parametrize(
    ("path", "bench", "width"),
    [
        ("src/tsim/statevector/schedule.py", True, True),
        ("src/tsim/statevector/shots.py", True, False),
        ("src/tsim/sampler.py", True, False),
        ("uv.lock", True, False),
        ("src/tsim/utils/diagram.py", False, False),
        ("src/tsim/noise/dem.py", False, False),
        ("docs/index.mdx", False, False),
    ],
)
def test_benchmarks_are_required_by_path_not_by_tier(path, bench, width):
    plan = review.classify([(10, 2, path)])
    assert (plan.bench, plan.width) == (bench, width)


def test_a_benchmark_with_a_regression_counts_as_run_and_is_noted():
    record = {"passed": True, "slower": 2, "wider": 0, "kept_differs": 1}
    assert review.regressions(record) == "2 slower, 1 with a different kept fraction"
    assert review.regressions({"passed": True, "slower": 0}) == ""


def test_tier_two_reviews_stop_after_three_rounds_with_findings():
    two = review.review_gaps(2, _rounds(("opus-5.5", 2), ("gpt-6-astra", 1)))
    assert len(two) == 1 and "new diff only" in two[0]
    three = review.review_gaps(
        2, _rounds(("opus-5.5", 2), ("gpt-6-astra", 1), ("opus-5.5", 1))
    )
    assert len(three) == 1 and "Stop reviewing" in three[0]


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        ("gh pr merge 281 --squash", True),
        ("uv run tsim-dev review status; gh pr merge 281", True),
        ("uv run tsim-dev review status && gh pr merge 281 --squash", False),
        ("gh pr view 281", False),
    ],
)
def test_the_bash_hook_lets_merges_through_the_review_gate_only(command, blocked):
    done = subprocess.run(  # noqa: S603
        [sys.executable, str(HOOK)],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": command}}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == (2 if blocked else 0)
