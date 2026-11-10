"""Static checks of the user skill in ``skills/``.

- Every ``SKILL.md`` follows the Agent Skills specification (https://agentskills.io/specification):
  frontmatter fields and their limits, and a ``name`` that matches its directory.
- Relative links in the skills point to files that exist.
- The Python code blocks of each file run, in order, in one namespace, as a reader would run
  them, and at least one of them asserts a result. A block that imports a package that is not
  installed or does not import (an optional decoder) is skipped.

Triggering and behavioral evals are not run in CI; see ``evals/skills/README.md``.
"""

import ast
import importlib
import re
from pathlib import Path

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
    missing = []
    for name in names:
        if name.split(".")[0] == "tsim":
            continue
        # NOTE: an optional decoder can be installed but fail to import; tesseract-decoder
        # ships py313-tagged wheels that pip and uv also install on Python 3.14.
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


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
def test_code_blocks_run(path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)  # blocks write figures and logs
    blocks = CODE_BLOCK.findall(path.read_text())
    # NOTE: blocks must check their results, not only run, so that a change in behavior fails.
    assert not blocks or any(
        "assert " in code for code in blocks
    ), "no block asserts a result"
    namespace: dict = {}
    for code in blocks:
        if _missing_modules(code):
            continue
        exec(compile(code, str(path), "exec"), namespace)  # noqa: S102
