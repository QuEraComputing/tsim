"""Run the Python code blocks of the user skills in ``skills/``.

The blocks of one file run in order in a shared namespace, as a reader would run them. A
block that imports a package that is not installed (an optional decoder) is skipped.
"""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
SKILL_FILES = sorted(SKILLS_DIR.glob("*/**/*.md"))
CODE_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _missing_modules(code: str) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return [n for n in names if importlib.util.find_spec(n.split(".")[0]) is None]


def test_skill_files_exist():
    assert {p.parent.name for p in SKILLS_DIR.glob("*/SKILL.md")} == {
        "tsim",
        "tsim-qec-study",
        "tsim-port-circuits",
    }


@pytest.mark.parametrize(
    "path", SKILL_FILES, ids=lambda p: str(p.relative_to(SKILLS_DIR))
)
def test_skill_code_blocks_run(path: Path):
    namespace: dict = {}
    for code in CODE_BLOCK.findall(path.read_text()):
        if _missing_modules(code):
            continue
        exec(compile(code, str(path), "exec"), namespace)  # noqa: S102


@pytest.mark.parametrize(
    "path", sorted(SKILLS_DIR.glob("*/SKILL.md")), ids=lambda p: p.parent.name
)
def test_skill_frontmatter(path: Path):
    text = path.read_text()
    assert text.startswith("---\n")
    front = text.split("---\n")[1]
    name = re.search(r"^name: (.+)$", front, re.MULTILINE)
    description = re.search(r"^description: (.+)$", front, re.MULTILINE)
    assert name is not None and name.group(1) == path.parent.name
    assert description is not None and len(description.group(1)) <= 1024
