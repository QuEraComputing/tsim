"""Mintlify docs build orchestrator.

Regenerates ``docs/api-reference/`` from ``src/tsim/`` (via griffe) and
``docs/tutorials/*.mdx`` from the source notebooks (via nbclient +
nbconvert). All generated files are gitignored.

Usage:
    python docs/build.py
    python docs/build.py --no-execute
    python docs/build.py --api-only
    python docs/build.py --notebooks-only
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import griffe

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
API_DIR = DOCS_DIR / "api-reference"
TUTORIALS_DIR = DOCS_DIR / "tutorials"
IMAGES_TUT_DIR = DOCS_DIR / "images" / "tutorials"
DOCS_JSON = REPO_ROOT / "docs.json"
SRC_DIR = REPO_ROOT / "src"


def clean(api: bool, notebooks: bool) -> None:
    """Remove generated output directories and files."""
    if api and API_DIR.exists():
        shutil.rmtree(API_DIR)
    if notebooks:
        if IMAGES_TUT_DIR.exists():
            shutil.rmtree(IMAGES_TUT_DIR)
        for mdx in TUTORIALS_DIR.glob("*.mdx"):
            mdx.unlink()


def _format_signature(obj) -> str:
    """Render a function or class signature as a Python source line."""
    name = obj.name
    params = []
    for p in obj.parameters:
        if p.name == "self":
            continue
        s = p.name
        if p.annotation:
            s += f": {p.annotation}"
        if p.default is not None and p.default != "":
            s += f" = {p.default}"
        params.append(s)
    sig = f"{name}({', '.join(params)})"
    if getattr(obj, "returns", None):
        sig += f" -> {obj.returns}"
    return sig


def _render_docstring(obj) -> str:
    """Return the parsed docstring as plain Markdown, or '' if none."""
    if not obj.docstring:
        return ""
    return obj.docstring.value.strip() + "\n"


def _render_member(member, level: int) -> list[str]:
    """Render a function/class member as MDX lines."""
    if member.name.startswith("_") and not member.name.startswith("__"):
        return []
    if member.kind == griffe.Kind.FUNCTION:
        return [
            f"{'#' * level} `{member.name}`",
            "",
            "```python",
            _format_signature(member),
            "```",
            "",
            _render_docstring(member),
            "",
        ]
    if member.kind == griffe.Kind.CLASS:
        out = [
            f"{'#' * level} class `{member.name}`",
            "",
            "```python",
            _format_signature(member),
            "```",
            "",
            _render_docstring(member),
            "",
        ]
        for sub_name, sub in sorted(member.members.items()):
            if sub_name.startswith("_") and sub_name != "__init__":
                continue
            if sub_name == "__init__":
                continue
            if sub.kind == griffe.Kind.FUNCTION:
                out.extend(_render_member(sub, level + 1))
        return out
    return []


def _render_module(module) -> str:
    """Render a single module to an MDX document."""
    title = module.path.split(".")[-1] or module.path
    desc = ""
    if module.docstring:
        first_line = module.docstring.value.strip().split("\n", 1)[0]
        desc = first_line.replace('"', "'")

    lines = [
        "---",
        f'title: "{title}"',
    ]
    if desc:
        lines.append(f'description: "{desc}"')
    lines.append("---")
    lines.append("")

    if module.docstring:
        lines.append(module.docstring.value.strip())
        lines.append("")

    classes = []
    functions = []
    for name, member in sorted(module.members.items()):
        if name.startswith("_"):
            continue
        if member.kind == griffe.Kind.CLASS:
            classes.append(member)
        elif member.kind == griffe.Kind.FUNCTION:
            functions.append(member)

    for c in classes:
        lines.extend(_render_member(c, level=2))
    for f in functions:
        lines.extend(_render_member(f, level=2))

    return "\n".join(lines).rstrip() + "\n"


def _module_output_path(dotted: str) -> Path:
    """Map ``tsim.core.parse`` to ``docs/api-reference/core/parse.mdx``.

    Package modules (``tsim`` itself, ``tsim.core``) become ``index.mdx`` in
    the corresponding directory.
    """
    parts = dotted.split(".")[1:]
    if not parts:
        return API_DIR / "index.mdx"
    return API_DIR / Path(*parts).with_suffix(".mdx")


def build_api() -> int:
    """Walk src/tsim/ with griffe and render one MDX per module."""
    pkg = griffe.load(
        "tsim",
        search_paths=[str(SRC_DIR)],
        extensions=griffe.load_extensions(
            "griffe_inherited_docstrings", "griffe_kirin"
        ),
    )
    API_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    visited: set[int] = set()

    def walk(module):
        nonlocal count
        # Guard against circular re-exports (griffe creates new objects per
        # import so track by object id, not by path string)
        obj_id = id(module)
        if obj_id in visited:
            return
        visited.add(obj_id)
        path = module.path
        # Only process modules that belong directly to the tsim source tree.
        # Re-exported packages like ``tsim.utils.encoder.tsim`` are noise.
        # A canonical tsim module path has no segment appearing more than once.
        segments = path.split(".")
        if len(segments) != len(set(segments)):
            return
        if not path.startswith("tsim"):
            return
        if "external" in path:
            return
        has_public = any(
            not n.startswith("_")
            and (m.kind == griffe.Kind.CLASS or m.kind == griffe.Kind.FUNCTION)
            for n, m in module.members.items()
        )
        if has_public or module.is_init_module:
            out = _module_output_path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(_render_module(module))
            count += 1
        for child in module.modules.values():
            walk(child)

    walk(pkg)
    return count


def build_notebooks(execute: bool) -> int:
    """Build tutorial MDX files from Jupyter notebooks."""
    print(f"[build_notebooks] stub — not implemented yet (execute={execute})")
    return 0


def main() -> int:
    """Run the docs build orchestrator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Skip kernel execution; render notebooks from cached outputs.",
    )
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--notebooks-only", action="store_true")
    args = parser.parse_args()

    if args.api_only and args.notebooks_only:
        parser.error("--api-only and --notebooks-only are mutually exclusive")

    do_api = not args.notebooks_only
    do_nb = not args.api_only

    start = time.perf_counter()
    clean(api=do_api, notebooks=do_nb)

    n_api = build_api() if do_api else 0
    n_nb = build_notebooks(execute=not args.no_execute) if do_nb else 0

    elapsed = time.perf_counter() - start
    print(f"\nBuilt {n_api} API modules, {n_nb} notebooks in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
