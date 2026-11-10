"""Tests for the development CLI in `dev/` and the agent hook in `.claude/`."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tsim_dev import bench, checks, cli, common

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / ".claude" / "hooks" / "guard_bash.py"


def test_registry_circuits_exist_and_the_quick_suite_is_small():
    registry = bench.registry()
    assert all(case.path.is_file() for case, _ in registry.values())
    assert all(
        case.sampler in ("detector", "measurement") for case, _ in registry.values()
    )
    assert all(case.postselect in ("all", "none") for case, _ in registry.values())
    quick = [name for name, (_, suite) in registry.items() if suite == "quick"]
    assert 4 <= len(quick) <= 8


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
        front = dict(
            line.split(": ", 1) for line in text.split("---\n")[1].strip().splitlines()
        )
        assert front["name"] == path.parent.name
        assert 0 < len(front["description"]) <= 1024
        assert len(text.splitlines()) < 500
