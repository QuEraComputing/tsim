"""Tests for the development CLI in `dev/` and the agent hook in `.claude/`."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from tsim_dev import bench, checks, cli, common, corpus, jobs, width

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


def test_benchmarks_time_bit_packed_rows_on_the_cpu_by_default(fake_bench):
    (case,) = bench.select(_bench_args())
    command = bench.worker_command(Path("python"), case, _bench_args(), "bench")
    assert "--bit-packed" in command
    assert command[command.index("--sampler") + 1] == "detector"
    command = bench.worker_command(
        Path("python"), case, _bench_args("--bools"), "bench"
    )
    assert "--bit-packed" not in command
    # NOTE: the recorded postselection is of detectors: a measurement sampler
    # samples every shot.
    (case,) = bench.select(_bench_args("--sampler", "measurement"))
    assert (case.sampler, case.postselect) == ("measurement", "none")
    side = bench.parse_side(".:device=gpu")
    assert side.describe().endswith("device=gpu")
    # NOTE: on the CPU the worker runs through on_device.py, which passes
    # device="cpu" to the samplers; on a GPU it runs as it is.
    command = bench.worker_command(Path("python"), case, _bench_args(), "bench")
    assert command[1:3] == [str(bench.ON_DEVICE), "cpu"]
    command = bench.worker_command(
        Path("python"), case, _bench_args("--device", "gpu"), "bench"
    )
    assert command[1].endswith("worker.py")


def test_shot_benchmarks_report_the_device_that_sampled(fake_bench, monkeypatch):
    devices = {"new": ["metal"] * 3, "ref": ["cpu"] * 3}

    def fake_measure(side, case, args, mode="bench"):
        result = {"shots_per_s": 1e6, "compile_s": 0.1, "acceptance": None, "shots": 1}
        result.update(active_width=6, device=devices[side.label].pop(0))
        return result

    monkeypatch.setattr(bench, "measure", fake_measure)
    (case,) = bench.select(_bench_args())
    sides = [bench.Side("new", Path("python")), bench.Side("ref", Path("python"))]
    args = _bench_args("--rounds", "2", "--thorough")
    args.rounds = 2
    record = bench.run_case(case, sides, args)
    assert (record["device"], record["ref_device"]) == ("metal", "cpu")
    assert "DEVICE: cpu -> metal" in bench.format_row(record, has_ref=True)
    devices.update(new=["metal", "metal", "cpu"], ref=["cpu"] * 3)
    assert "device changed between runs" in bench.run_case(case, sides, args)["error"]


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
        ({"command": "pkill --full fuzz.py"}, True),
        ({"command": "pkill -9 --full fuzz"}, True),
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
    assert common.child_env(side.threads)["NUMBA_NUM_THREADS"] == "10"
    # A python executable is used as given: resolving a venv's symlink would lose the venv.
    python = Path(sys.executable).absolute()
    assert bench.parse_side(f"{python}:threads=2").python == python
    for bad in (".:cores=2", ".:threads=0", ".:device=tpu", ".:threads"):
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


def _fake_width_stream(calls, stop):
    def fake_stream(cmd, env, case_timeout, deadline, outcome):
        cases = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--case"]
        calls.append(cases)
        for case in cases:
            yield f"START {case}"
            if (case, len(calls)) in stop:
                setattr(outcome, stop[case, len(calls)], True)
                return
            yield "RESULT " + json.dumps({"name": case, "width": 3})

    return fake_stream


def _measure(cases, args):
    results = {}
    width.measure(
        bench.parse_side("."), cases, args, time.monotonic() + 100, results.__setitem__
    )
    return results


def test_width_mode_resumes_after_a_stuck_circuit(fake_bench, monkeypatch):
    calls = []
    stop = {("p/narrow", 1): "timed_out"}
    monkeypatch.setattr(width, "_stream", _fake_width_stream(calls, stop))
    cases = bench.select(_bench_args("--width", "--suite", "corpus"))
    results = _measure(cases, _bench_args("--width"))
    assert results["p/narrow"]["error"].startswith("compile took over")
    assert results["p/wide"] == {"width": 3}
    assert calls == [["p/narrow", "p/wide"], ["p/wide"]]


def test_width_mode_stops_at_the_overall_deadline(fake_bench, monkeypatch):
    calls = []
    stop = {("p/narrow", 1): "past_deadline"}
    monkeypatch.setattr(width, "_stream", _fake_width_stream(calls, stop))
    cases = bench.select(_bench_args("--width", "--suite", "corpus"))
    args = _bench_args("--width", "--timeout", "60")
    results = _measure(cases, args)
    assert all("--timeout 60" in r["error"] for r in results.values())
    assert calls == [["p/narrow", "p/wide"]]


def test_width_results_stream_from_both_sides(fake_bench, monkeypatch):
    monkeypatch.setattr(width, "_stream", _fake_width_stream([], {}))
    cases = bench.select(_bench_args("--width", "--suite", "corpus"))
    sides = [bench.parse_side(".:threads=1"), bench.parse_side(".:threads=2")]
    args = _bench_args("--width", "--jobs", "4")
    got = list(width.measure_all(sides, cases, args, time.monotonic() + 100))
    assert sorted((i, k) for i, k, _ in got) == [
        (0, "p/narrow"),
        (0, "p/wide"),
        (1, "p/narrow"),
        (1, "p/wide"),
    ]


def test_the_width_stream_stops_a_stuck_compile():
    script = "import time; print('START a', flush=True); time.sleep(30)"
    outcome = width.Exit()
    command = [sys.executable, "-c", script]
    lines = list(width._stream(command, None, 1, time.monotonic() + 100, outcome))
    assert lines == ["START a"] and outcome.timed_out and not outcome.past_deadline


def test_shot_benchmarks_compare_and_check_the_active_width(fake_bench, monkeypatch):
    widths = {"new": [6, 6, 6], "ref": [6, 6, 6]}

    def fake_measure(side, case, args, mode="bench"):
        result = {"shots_per_s": 1e6, "compile_s": 0.1, "acceptance": None, "shots": 1}
        result["active_width"] = widths[side.label].pop(0)
        return result

    monkeypatch.setattr(bench, "measure", fake_measure)
    (case,) = bench.select(_bench_args())
    sides = [bench.Side("new", Path("python")), bench.Side("ref", Path("python"))]
    args = _bench_args("--rounds", "2", "--thorough")
    args.rounds = 2
    record = bench.run_case(case, sides, args)
    assert (record["k"], record["ref_k"], record["width"]) == (6, 6, "same")

    widths.update(new=[7, 7, 7], ref=[6, 6, 6])
    record = bench.run_case(case, sides, args)
    assert record["width"] == "wider"
    assert "WIDER: k 6 -> 7" in bench.format_row(record, has_ref=True)

    widths.update(new=[6, 6, 7], ref=[6, 6, 6])
    assert "changed between runs" in bench.run_case(case, sides, args)["error"]


def test_deleted_files_still_choose_tests():
    # A deleted source file can break its importers: run everything.
    assert checks.tests_for(["src/tsim/no_longer_here.py"]) is None
    # A deleted test file has nothing left to run.
    assert checks.tests_for(["test/unit/test_no_longer_here.py"]) == []


def test_sides_with_the_same_label_are_told_apart(tmp_path):
    one = tmp_path / "a" / ".venv" / "bin" / "python"
    two = tmp_path / "b" / ".venv" / "bin" / "python"
    sides = bench.distinct(
        [bench.Side("main@1234-dirty", one), bench.Side("main@1234-dirty", two)]
    )
    assert sides[0].describe() != sides[1].describe()
    assert str(tmp_path / "a") in sides[0].label
    with pytest.raises(common.InfraError, match="the same"):
        bench.distinct([bench.Side("x", one), bench.Side("x", one)])


def test_run_maps_command_failures_to_exit_one(capsys):
    args = cli.build_parser().parse_args(["run"])
    args.command = [sys.executable, "-c", "raise SystemExit(42)"]
    assert jobs.run(args) == 1
    assert "exit code 42" in capsys.readouterr().out
    args.command = [sys.executable, "-c", "pass"]
    assert jobs.run(args) == 0


@pytest.mark.parametrize(
    ("new", "ref", "done"),
    [
        ([1.0, 1.01, 0.99], [1.0, 1.0, 1.0], True),  # all within the tolerance
        ([1.3, 1.2, 1.25], [1.0, 1.0, 1.0], True),  # all faster
        ([1.0, 1.2, 0.99], [1.0, 1.0, 1.0], False),  # one round beyond it
        ([1.2, 0.9, 1.2], [1.0, 1.0, 1.0], False),  # noisy
    ],
)
def test_rounds_stop_once_more_could_not_change_the_verdict(new, ref, done):
    assert bench.decided(new, ref, tolerance=0.05) is done


def test_served_rounds_share_one_call_and_stop_early(fake_bench, monkeypatch):
    rates = {"new": [], "ref": []}
    requests, starts = [], []

    class FakeWorker:
        def __init__(self, side, case, args):
            self.label = side.label
            starts.append(side.label)
            self.ready = {}

        def wait_ready(self):
            # NOTE: the new side's first start compiles a kernel (a cold cache).
            self.ready = {
                "compile_s": 0.1,
                "cache_misses": int(self.label == "new" and len(starts) <= 6),
            }
            return self.ready

        def request(self, line, answer):
            requests.append((self.label, line))
            if answer == "SHOTS":
                return {"shots": 1000, "calls": 2}
            result = {"shots_per_s": rates[self.label].pop(0), "compile_s": 0.1}
            result.update(acceptance=None, shots=2000, active_width=4, device="cpu")
            return result

        def close(self):
            pass

    def fake_measure(side, case, args, mode="bench"):
        return {"compile_s": 0.1}

    monkeypatch.setattr(bench, "Worker", FakeWorker)
    monkeypatch.setattr(bench, "measure", fake_measure)
    (case,) = bench.select(_bench_args())
    sides = [bench.Side("new", Path("python")), bench.Side("ref", Path("python"))]
    args = _bench_args()
    args.rounds, args.min_seconds = 5, 0.3
    rates.update(new=[1e6] * 5, ref=[1e6] * 5)
    record = bench.run_case(case, sides, args)
    assert len(record["rounds"]["new"]) == 3
    assert record["rate"]["verdict"] == "same"
    # NOTE: the first side calibrates; both sides time the same calls.
    assert requests[0] == ("new", "calibrate 0.3")
    assert {line for _, line in requests[1:]} == {"time 1000 2 2"}
    # NOTE: the first rounds' pairs start at once; a worker that compiled a kernel
    # during its warmup restarts the batch.
    assert starts == ["new", "ref"] * 6
    rates.update(new=[1.2e6, 0.9e6, 1.2e6, 1.0e6, 1.0e6], ref=[1e6] * 5)
    starts.clear()
    record = bench.run_case(case, sides, args)
    assert len(record["rounds"]["new"]) == 5
    # NOTE: every round has a pair of its own; rounds past the first batch start one.
    assert len(starts) == 2 * (6 + 2)
    # NOTE: a side on another device calibrates its own call.
    sides[0] = bench.Side("new", Path("python"), device="gpu")
    rates.update(new=[1e6] * 5, ref=[1e6] * 5)
    requests.clear()
    bench.run_case(case, sides, args)
    assert requests[:2] == [("new", "calibrate 0.3"), ("ref", "calibrate 0.3")]
