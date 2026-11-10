"""Static checks of the user skills in ``skills/``.

- Every ``SKILL.md`` follows the Agent Skills specification (https://agentskills.io/specification):
  frontmatter fields and their limits, and a ``name`` that matches its directory.
- Relative links in the skills point to files that exist.
- The Python code blocks of each file run, in order, in one namespace, as a reader would run
  them. A block that imports a package that is not installed (an optional decoder) is skipped.

Triggering and behavioral evals are not run in CI; see ``evals/skills/README.md``.
"""

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
SKILL_DIRS = sorted(p.parent for p in SKILLS_DIR.glob("*/SKILL.md"))
MARKDOWN_FILES = sorted(SKILLS_DIR.glob("*/**/*.md"))
CODE_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
LINK = re.compile(r"\]\(([^)\s]+)\)")
NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
ALLOWED_FIELDS = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}


def _ids(path: Path) -> str:
    return str(path.relative_to(SKILLS_DIR))


def _frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text()
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    head, body = text[4:].split("\n---\n", 1)
    front = yaml.safe_load(head)
    assert isinstance(front, dict)
    return front, body


def _missing_modules(code: str) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return [n for n in names if importlib.util.find_spec(n.split(".")[0]) is None]


def test_expected_skills_exist():
    assert {p.name for p in SKILL_DIRS} == {
        "tsim",
        "tsim-qec-experiment",
        "tsim-port-circuits",
    }


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda p: p.name)
def test_frontmatter_follows_the_spec(skill_dir: Path):
    front, body = _frontmatter(skill_dir / "SKILL.md")
    assert set(front) <= ALLOWED_FIELDS, f"unknown fields {set(front) - ALLOWED_FIELDS}"
    name = front["name"]
    assert isinstance(name, str) and len(name) <= 64 and NAME.match(name)
    assert name == skill_dir.name
    description = front["description"]
    assert isinstance(description, str) and 1 <= len(description) <= 1024
    assert len(str(front.get("compatibility", ""))) <= 500
    metadata = front.get("metadata", {})
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())
    # NOTE: the spec recommends keeping SKILL.md under 500 lines; details go in references/.
    assert len(body.splitlines()) < 500


@pytest.mark.parametrize("path", MARKDOWN_FILES, ids=_ids)
def test_relative_links_resolve(path: Path):
    for target in LINK.findall(path.read_text()):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        assert (path.parent / target.split("#")[0]).exists(), f"broken link {target}"


@pytest.mark.parametrize("path", MARKDOWN_FILES, ids=_ids)
def test_code_blocks_run(path: Path):
    namespace: dict = {}
    for code in CODE_BLOCK.findall(path.read_text()):
        if _missing_modules(code):
            continue
        exec(compile(code, str(path), "exec"), namespace)  # noqa: S102


CHECK_MASK = SKILLS_DIR / "tsim" / "scripts" / "check_mask.py"
CIRCUIT = """
R 0 1 2
X 0
X_ERROR(0.01) 0 1 2
H 2
M 0 1 2
DETECTOR(0, 0, 0) rec[-3]
DETECTOR(1, 0, 0) rec[-2]
DETECTOR(2, 0, 1) rec[-1]
"""


def _check_mask(tmp_path: Path, plan: dict | None, *args: str):
    circuit = tmp_path / "c.stim"
    circuit.write_text(CIRCUIT)
    cmd = [sys.executable, str(CHECK_MASK), str(circuit)]
    if plan is not None:
        (tmp_path / "plan.json").write_text(json.dumps(plan))
        cmd.append(str(tmp_path / "plan.json"))
    cmd += ["--shots", "1000", "--noiseless-shots", "200", *args]
    # The command runs the skill's own script with this interpreter.
    return subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, timeout=300
    )


def test_check_mask_summary(tmp_path: Path):
    result = _check_mask(tmp_path, None, "--summary")
    assert result.returncode == 0, result.stderr
    assert "D0-D1 (2): time coordinate 0" in result.stdout


def test_check_mask_accepts_a_good_plan(tmp_path: Path):
    plan = {
        "stages": [
            {"name": "checks", "detectors": "1"},
            {"name": "flag", "detectors": [0], "action": "ignore"},
            {"name": "readout", "detectors": "2-2", "action": "ignore"},
        ]
    }
    result = _check_mask(tmp_path, plan, "--out", str(tmp_path / "mask.npy"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert np.load(tmp_path / "mask.npy").tolist() == [False, True, False]


def test_check_mask_rejects_a_bad_plan(tmp_path: Path):
    plan = {
        "stages": [{"name": "all", "detectors": "0-1"}, {"name": "x", "detectors": [1]}]
    }
    result = _check_mask(tmp_path, plan)
    assert result.returncode == 1
    out = result.stdout
    assert "repeats detectors already in a stage: 1" in out
    assert "detectors in no stage: 2" in out
    assert "postselected detectors 0 are 1 in every noiseless shot" in out
