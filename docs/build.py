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
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import griffe
import nbformat
from nbclient import NotebookClient
from nbconvert.exporters import MarkdownExporter
from nbconvert.preprocessors import (
    ExtractOutputPreprocessor,
    TagRemovePreprocessor,
)

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


def _escape_mdx_text(text: str) -> str:
    """Escape characters in Markdown text that MDX parses as JSX/expressions.

    MDX treats ``<``, ``>``, ``{``, and ``}`` as JSX/expression delimiters
    when they appear outside fenced code blocks. Docstrings often contain
    these in physics notation (|0>, ->, <->, sum_{...}) which trips the
    parser. We escape them so Mintlify renders them as plain text.

    Only processes lines that are outside fenced code blocks (``` fences).
    Inside fences Mintlify renders content verbatim — no escaping needed.
    """
    lines = text.split("\n")
    result: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            result.append(line)
            continue
        if in_fence:
            result.append(line)
        else:
            # Escape characters MDX parses as JSX/expression syntax.
            escaped = (
                line.replace("{", r"\{")
                .replace("}", r"\}")
                .replace("<", r"\<")
                .replace(">", r"\>")
            )
            result.append(escaped)
    return "\n".join(result)


def _extract_diagram_svgs(
    body: str, img_dir: Path, url_prefix: str
) -> tuple[str, list[tuple[Path, bytes]]]:
    """Replace tsim diagram HTML with Markdown image references.

    tsim's ``Diagram._repr_html_()`` wraps the SVG in a zoom container div
    plus a ``<script>`` block. MDX cannot parse the ``<script>`` tags and has
    trouble with SVG elements like ``<text>`` and ``<tspan>`` in inline
    context.

    This function extracts each embedded SVG, saves it to *img_dir* as a
    standalone ``.svg`` file, and replaces the whole HTML block with a
    Markdown image reference ``![Circuit diagram](/url_prefix/NNN.svg)``.

    Returns the modified body string and a list of ``(path, data)`` tuples
    for files that need to be written to disk.
    """
    tsim_pattern = re.compile(
        r'<div\s+data-tsim-zoom="[^"]*"[^>]*>.*?(<svg\b[^>]*>.*?</svg>)'
        r"\s*</div>\s*</div>\s*</div>"
        r"(?:\s*<script\b[^>]*>.*?</script>)?",
        re.DOTALL | re.IGNORECASE,
    )

    files_to_write: list[tuple[Path, bytes]] = []
    counter = [0]

    def replacer(m: re.Match) -> str:
        counter[0] += 1
        svg_content = m.group(1)
        fname = f"diagram_{counter[0]:03d}.svg"
        files_to_write.append((img_dir / fname, svg_content.encode("utf-8")))
        return f"\n\n![Circuit diagram]({url_prefix}/{fname})\n\n"

    body = tsim_pattern.sub(replacer, body)
    return body, files_to_write


def _strip_script_and_pyzx(body: str) -> str:
    """Remove pyzx D3 script blocks and any remaining <script> tags.

    pyzx emits a ``<div id="graph-output-…">`` placeholder followed by a
    large ``<script type="module">`` block that renders an interactive ZX
    diagram using D3. Both the script and the empty div are useless in a
    static docs context, so we drop the script and leave the (now empty)
    placeholder div in place to avoid breaking paragraph flow.

    Any remaining ``<script>`` tags from other sources are also stripped.
    """
    return re.sub(
        r"<script\b[^>]*>.*?</script>",
        "",
        body,
        flags=re.DOTALL | re.IGNORECASE,
    )


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
    """Return the parsed docstring as MDX-safe Markdown, or '' if none."""
    if not obj.docstring:
        return ""
    return _escape_mdx_text(obj.docstring.value.strip()) + "\n"


def _render_member(member, level: int) -> list[str]:
    """Render a function/class member as MDX lines."""
    if member.name.startswith("_") and not member.name.startswith("__"):
        return []
    # NOTE: use .kind (not .is_class / .is_function) — Alias.kind safely
    # returns Kind.ALIAS on unresolvable imports; the is_* properties raise.
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
        lines.append(_escape_mdx_text(module.docstring.value.strip()))
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


def _module_output_path(module) -> Path:
    """Map module to its output MDX path.

    - Package ``__init__`` modules → ``<subdir>/index.mdx`` (the landing
      page for that subpackage).
    - Regular submodules → ``<a>/<b>.mdx``.
    - The top-level ``tsim`` package → ``api-reference/index.mdx``.
    """
    dotted = module.path
    parts = dotted.split(".")[1:]
    if not parts:
        return API_DIR / "index.mdx"
    if module.is_init_module:
        return API_DIR / Path(*parts) / "index.mdx"
    return API_DIR / Path(*parts).with_suffix(".mdx")


def _build_api_groups(rendered_paths: list[Path]) -> list[dict]:
    """Map rendered MDX paths to Mintlify navigation groups.

    Each top-level subdirectory of api-reference/ becomes one group;
    files directly under api-reference/ go into a "Top level" group.
    Inside each group, the subpackage ``index`` page is listed first
    (it's the landing page) and the remaining modules follow alphabetically.
    """
    top_level: list[str] = []
    by_dir: dict[str, list[str]] = {}

    for p in rendered_paths:
        rel = p.relative_to(DOCS_DIR).with_suffix("").as_posix()
        # rel looks like "api-reference/core/parse" or "api-reference/circuit"
        parts = rel.split("/")
        if len(parts) == 2:
            top_level.append(rel)
        else:
            subpkg = parts[1]
            by_dir.setdefault(subpkg, []).append(rel)

    def _index_first(pages: list[str]) -> list[str]:
        """Return pages with any ``.../index`` entry sorted to the front."""
        indexes = sorted(
            p for p in pages if p.endswith("/index") or p == "api-reference/index"
        )
        others = sorted(p for p in pages if p not in indexes)
        return indexes + others

    groups: list[dict] = []
    if top_level:
        groups.append({"group": "Top level", "pages": _index_first(top_level)})
    for subpkg in sorted(by_dir):
        groups.append({"group": subpkg, "pages": _index_first(by_dir[subpkg])})
    return groups


def _patch_docs_json(api_groups: list[dict]) -> None:
    """Overwrite the "API reference" tab's groups inside docs.json.

    Gracefully no-ops if docs.json doesn't exist yet (Task 5 creates it).
    """
    if not DOCS_JSON.exists():
        print(f"[warn] {DOCS_JSON} not found; skipping nav patch")
        return
    cfg = json.loads(DOCS_JSON.read_text())
    tabs = cfg.get("navigation", {}).get("tabs", [])
    for tab in tabs:
        if tab.get("tab") == "API reference":
            tab["groups"] = api_groups
            break
    else:
        print("[warn] no 'API reference' tab in docs.json; skipping nav patch")
        return
    DOCS_JSON.write_text(json.dumps(cfg, indent=2) + "\n")


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
    rendered: list[Path] = []
    visited_ids: set[int] = set()
    visited_paths: set[str] = set()

    def walk(module):
        # rendered is closed over
        # Guard against circular re-exports. Griffe creates new objects for
        # each level of the phantom re-export (e.g. ``import tsim`` inside
        # encoder.py produces ``tsim.utils.encoder.tsim``, then
        # ``tsim.utils.encoder.tsim.utils.encoder.tsim``, etc.), so object-id
        # tracking alone is insufficient — the ids are always fresh. Track
        # both ids and paths as a belt-and-suspenders guard, but also detect
        # phantom cycles by checking whether any prefix sub-path of the
        # current path was already visited: a phantom path like
        # ``tsim.utils.encoder.tsim`` re-introduces ``tsim`` (a visited root)
        # as a non-root component, so its dot-segments will contain a visited
        # path as a suffix component.
        obj_id = id(module)
        if obj_id in visited_ids:
            return
        visited_ids.add(obj_id)
        path = module.path
        if path in visited_paths:
            return
        # Detect phantom re-export cycles: if any proper suffix of the path
        # (as dot-segments) matches an already-visited path, this is a
        # circular re-export, not a real new module.
        segments = path.split(".")
        for i in range(1, len(segments)):
            if ".".join(segments[i:]) in visited_paths:
                return
        visited_paths.add(path)
        # Only process modules that belong directly to the tsim source tree.
        # Re-exported packages like ``tsim.utils.encoder.tsim`` are noise.
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
            out = _module_output_path(module)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(_render_module(module))
            rendered.append(out)
        for child in module.modules.values():
            walk(child)

    walk(pkg)
    _patch_docs_json(_build_api_groups(rendered))
    return len(rendered)


def _humanize(name: str) -> str:
    """Convert a snake_case or kebab-case filename stem to a title string."""
    return name.replace("_", " ").replace("-", " ").capitalize()


def _extract_title(nb: Any, fallback_name: str) -> tuple[str, Any]:
    """Pull title from first markdown cell if it starts with '# '.

    Strip that line from the cell to avoid a duplicate H1 in the
    rendered MDX. Returns (title, possibly-mutated-notebook).
    """
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            source = cell.source.lstrip()
            m = re.match(r"^#\s+([^\n]+)", source)
            if m:
                title = m.group(1).strip()
                cell.source = source[m.end() :].lstrip("\n")
                return title, nb
            break  # first md cell doesn't have an H1; stop scanning
    return _humanize(fallback_name), nb


def _convert_notebook(nb_path: Path, execute: bool) -> None:
    """Convert a single .ipynb to .mdx, saving extracted figures alongside."""
    nb = nbformat.read(str(nb_path), as_version=4)

    if execute:
        # Add docs/demos to PYTHONPATH so notebooks that import local helpers
        # (e.g. ``from utils.no_decoder import …``) can find them at execution
        # time without modifying the notebook sources.
        demos_dir = str(DOCS_DIR / "demos")
        old_pythonpath = os.environ.get("PYTHONPATH")
        existing = old_pythonpath or ""
        os.environ["PYTHONPATH"] = f"{demos_dir}:{existing}".rstrip(":")
        try:
            client = NotebookClient(
                nb,
                timeout=600,
                kernel_name="python3",
                allow_errors=False,
                resources={"metadata": {"path": str(nb_path.parent)}},
            )
            client.execute()
        finally:
            if old_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_pythonpath

    # Drop cells tagged "remove-cell" (post-execution).
    tag_remove = TagRemovePreprocessor(remove_cell_tags=("remove-cell",))
    nb, _ = tag_remove.preprocess(nb, {})

    title, nb = _extract_title(nb, nb_path.stem)

    out_img_dir = IMAGES_TUT_DIR / nb_path.stem
    out_img_dir.mkdir(parents=True, exist_ok=True)
    extract = ExtractOutputPreprocessor()
    extract.output_filename_template = "cell_{cell_index}_{index}{extension}"

    exporter = MarkdownExporter()
    exporter.register_preprocessor(extract, enabled=True)
    body, resources = exporter.from_notebook_node(
        nb,
        resources={
            "output_files_dir": f"/images/tutorials/{nb_path.stem}",
            "unique_key": nb_path.stem,
        },
    )

    # Save extracted resources (matplotlib figures, etc.)
    for fname, data in resources.get("outputs", {}).items():
        out_file = out_img_dir / Path(fname).name
        out_file.write_bytes(data)

    # Replace tsim diagram HTML wrappers with Markdown image refs: extracts
    # each SVG, saves it as a standalone file, and inserts ![alt](url).
    url_prefix = f"/images/tutorials/{nb_path.stem}"
    body, extra_files = _extract_diagram_svgs(body, out_img_dir, url_prefix)
    for fpath, data in extra_files:
        fpath.write_bytes(data)

    # Strip pyzx D3 <script> blocks and any other remaining <script> tags.
    body = _strip_script_and_pyzx(body)

    # Replace bare "![svg](...)" alt text (nbconvert's default) with something
    # more descriptive for accessibility.
    body = re.sub(
        r"!\[svg\]\(",
        "![Notebook output figure](",
        body,
    )

    title_esc = title.replace('"', "'")
    mdx = f'---\ntitle: "{title_esc}"\n---\n\n{body.rstrip()}\n'
    out_mdx = nb_path.with_suffix(".mdx")
    out_mdx.write_text(mdx)


def build_notebooks(execute: bool) -> int:
    """Build tutorial MDX files from Jupyter notebooks."""
    if not TUTORIALS_DIR.exists():
        return 0
    notebooks = sorted(TUTORIALS_DIR.glob("*.ipynb"))
    for nb_path in notebooks:
        _convert_notebook(nb_path, execute=execute)
    return len(notebooks)


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
