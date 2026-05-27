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
DOCS_JSON = DOCS_DIR / "docs.json"
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


def _stretch_zoom_block(block: str) -> str:
    """Make a tsim zoom container fill the iframe's full width.

    In Jupyter the container is sized to the circuit, which inside a
    full-width iframe leaves a tiny bordered panel floating in a large
    empty frame. Rewrite the outer ``data-tsim-zoom`` container to fill the
    iframe (``width`` and ``height`` 100%, ``box-sizing:border-box``). The
    container's child ``size`` div is ``display:inline-block``, so the line
    box reserves room for a font descender below it (a few px), which pushed
    content past the frame and produced a spurious vertical scrollbar; set
    ``font-size:0; line-height:0`` on the container to remove that gap. The
    rounded border lives on the iframe itself, so drop the container border
    here. Also drop the script's auto-fit-to-width step so the diagram keeps
    its height-fit initial scale: small circuits render at natural size
    (left-aligned) and wide circuits keep full height and scroll
    horizontally, instead of being squeezed thin to fit the width.
    """
    block = re.sub(
        r'(<div\b[^>]*data-tsim-zoom="[^"]*"[^>]*style=")[^"]*"',
        r"\1width:100%; height:100%; box-sizing:border-box; "
        r"font-size:0; line-height:0; overflow:auto; background:white; "
        r'border:none; position:relative;"',
        block,
        count=1,
    )
    block = block.replace(
        "scale = cw / natW;",
        "/* keep height-fit initial scale */",
    )
    return block


def _extract_diagram_svgs(
    body: str, img_dir: Path, url_prefix: str
) -> tuple[str, list[tuple[Path, bytes]]]:
    """Replace tsim diagram HTML with iframe embeds.

    tsim's ``Diagram._repr_html_()`` wraps the SVG in a zoom container div
    plus a ``<script>`` block. MDX cannot parse ``<script>`` tags or the
    inline ``style="..."`` attributes from raw HTML, so we extract the full
    interactive block into a standalone HTML file and embed it via an
    ``<iframe>``. This preserves pan/zoom that the Jupyter renderer
    provides.

    Returns the modified body string and a list of ``(path, data)`` tuples
    for files to write to disk.
    """
    tsim_pattern = re.compile(
        r'<div\s+data-tsim-zoom="[^"]*"[^>]*'
        r"height:(?P<height>\d+(?:\.\d+)?)px[^>]*>.*?</div>\s*</div>\s*</div>"
        r"(?:\s*<script\b[^>]*>.*?</script>)?",
        re.DOTALL | re.IGNORECASE,
    )

    files_to_write: list[tuple[Path, bytes]] = []
    counter = [0]

    def replacer(m: re.Match) -> str:
        counter[0] += 1
        block = _stretch_zoom_block(m.group(0))
        # Replace the random uuid (used in both the container attribute and
        # the script's querySelector) with a deterministic per-notebook id so
        # the generated file is byte-stable across builds (avoids git churn).
        uid_m = re.search(r'data-tsim-zoom="([^"]+)"', block)
        if uid_m:
            block = block.replace(uid_m.group(1), f"tsim-zoom-{counter[0]:03d}")
        height = int(float(m.group("height")))
        # body overflow:hidden so only the inner zoom container scrolls
        # (otherwise the iframe body scrolls too -> nested scrollbars).
        html = (
            '<!doctype html><html><head><meta charset="utf-8">'
            "<style>html,body{margin:0;padding:0;height:100%;"
            "overflow:hidden;}</style>"
            f"</head><body>{block}</body></html>"
        )
        fname = f"diagram_{counter[0]:03d}.html"
        files_to_write.append((img_dir / fname, html.encode("utf-8")))
        src = f"{url_prefix}/{fname}"
        return (
            f'\n\n<iframe src="{src}" width="100%" height="{height + 4}" '
            f'style={{{{border: "1px solid #eee", borderRadius: "0.5rem"}}}} '
            f'title="Circuit diagram {counter[0]}" loading="lazy" />\n\n'
        )

    body = tsim_pattern.sub(replacer, body)
    return body, files_to_write


def _normalize_block_math(body: str) -> str:
    """Put each ``$$`` delimiter on its own line.

    Mintlify's MDX parser only recognizes a ``$$...$$`` block as display math
    when both delimiters sit on their own line. Notebooks frequently write
    multi-line math with the opening (and sometimes closing) ``$$`` glued to
    text on the same line, which breaks rendering and bleeds raw LaTeX into
    the page. Split such delimiters onto their own lines, skipping content
    inside fenced code blocks.
    """
    lines = body.split("\n")
    out: list[str] = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or "$$" not in line:
            out.append(line)
            continue
        # Count $$ occurrences; if exactly 2 on one line it's inline display
        # math, leave it alone. Otherwise (1 occurrence = open or close of a
        # multi-line block), isolate it.
        if line.count("$$") == 1:
            idx = line.index("$$")
            before = line[:idx].rstrip()
            after = line[idx + 2 :].lstrip()
            if before:
                out.append(before)
            out.append("$$")
            if after:
                out.append(after)
        else:
            out.append(line)
    return "\n".join(out)


def _extract_pyzx_iframes(
    body: str, img_dir: Path, url_prefix: str
) -> tuple[str, list[tuple[Path, bytes]]]:
    """Replace pyzx D3 diagram blocks with iframe embeds.

    ``c.diagram("pyzx")`` emits a ``<div id="graph-output-…">`` placeholder
    followed by a ``<script type="module">`` that imports D3 from a CDN and
    renders an interactive ZX diagram into the div. MDX cannot parse the
    ``<script>`` tag, but a standalone HTML page can host both, so we save
    each pair to its own ``.html`` file and embed it via ``<iframe>``.
    """
    pattern = re.compile(
        r'(<div\b[^>]*id="graph-output-[^"]*"[^>]*>\s*</div>)\s*'
        r'(<script\b[^>]*type="module"[^>]*>.*?</script>)',
        re.DOTALL | re.IGNORECASE,
    )
    files_to_write: list[tuple[Path, bytes]] = []
    counter = [0]

    # The D3 render call ends with ``…, <width>, <height>, <pad>, <scale>,
    # <bool>, <bool>, '');`` — pull the SVG height so the iframe matches the
    # diagram instead of using a fixed height that leaves large white margins.
    height_re = re.compile(
        r",\s*[\d.]+,\s*([\d.]+),\s*[\d.]+,\s*[\d.]+,\s*"
        r"(?:true|false),\s*(?:true|false)",
        re.IGNORECASE,
    )

    def replacer(m: re.Match) -> str:
        counter[0] += 1
        script = m.group(2)
        block = m.group(1) + script
        # Replace pyzx's random ``graph-output-XXXX`` id (used in the div and
        # the ``showGraph('#…')`` call) with a deterministic per-notebook id
        # so the generated file is byte-stable across builds. The CSS uses an
        # ``[id^="graph-output-"]`` prefix selector, so it still matches.
        gid_m = re.search(r'id="(graph-output-[^"]+)"', block)
        if gid_m:
            block = block.replace(gid_m.group(1), f"graph-output-{counter[0]:03d}")
        hm = height_re.search(script)
        height = int(float(hm.group(1))) + 12 if hm else 400
        html = (
            '<!doctype html><html><head><meta charset="utf-8">'
            "<style>html,body{margin:0;padding:0;height:100%;"
            "overflow:hidden;}"
            '[id^="graph-output-"]{width:100%;height:100%;'
            "overflow:auto;box-sizing:border-box;}</style>"
            f"</head><body>{block}</body></html>"
        )
        fname = f"pyzx_{counter[0]:03d}.html"
        files_to_write.append((img_dir / fname, html.encode("utf-8")))
        src = f"{url_prefix}/{fname}"
        return (
            f'\n\n<iframe src="{src}" width="100%" height="{height}" '
            f'style={{{{border: "1px solid #eee", borderRadius: "0.5rem"}}}} '
            f'title="ZX diagram {counter[0]}" loading="lazy" />\n\n'
        )

    body = pattern.sub(replacer, body)
    return body, files_to_write


def _strip_script_and_pyzx(body: str) -> str:
    """Strip any remaining ``<script>`` tags and empty graph placeholder divs.

    Runs after :func:`_extract_pyzx_iframes`, which captures real pyzx
    output pairs. Anything left over here is either an orphaned script or
    an empty placeholder (no following script) that would crash MDX on the
    inline ``style="..."`` attribute.
    """
    body = re.sub(
        r"<script\b[^>]*>.*?</script>",
        "",
        body,
        flags=re.DOTALL | re.IGNORECASE,
    )
    body = re.sub(
        r'<div\b[^>]*id="graph-output-[^"]*"[^>]*>\s*</div>',
        "",
        body,
        flags=re.IGNORECASE,
    )
    return body


def _fence_indented_outputs(body: str) -> str:
    """Convert nbconvert's 4-space indented output blocks to fenced blocks.

    nbconvert's ``MarkdownExporter`` emits cell stream/text outputs as
    indented (4-space) code blocks. Mintlify's MDX parser does not
    consistently recognize indented code blocks, which causes the output
    to render as a normal paragraph — whitespace and newlines collapse,
    and any literal escape sequences in the printed text show up
    untouched. Rewrap each contiguous indented output as a fenced
    ```` ```text title="Output" ```` block so it renders correctly and is
    visually distinct from input code cells.
    """
    lines = body.split("\n")
    out: list[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence or not line.startswith("    "):
            out.append(line)
            i += 1
            continue
        # Start of an indented block. Collect until a non-indented,
        # non-blank line. Allow blank lines inside the block.
        block: list[str] = []
        while i < len(lines):
            cur = lines[i]
            if cur.startswith("    "):
                block.append(cur[4:])
                i += 1
            elif cur.strip() == "":
                # Look ahead: if the next non-empty line is still
                # indented, the blank is part of the block.
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines) and lines[j].startswith("    "):
                    block.append("")
                    i += 1
                else:
                    break
            else:
                break
        # Trim trailing blank lines inside the block.
        while block and block[-1] == "":
            block.pop()
        if block:
            out.append('```text title="Output"')
            out.extend(block)
            out.append("```")
            out.append("")
    return "\n".join(out)


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


def _one_line(text: str) -> str:
    """Collapse internal whitespace so text fits on a single list item."""
    return " ".join(text.split())


def _render_docstring_sections(sections) -> str:
    """Render parsed griffe docstring sections to MDX-safe Markdown."""
    out: list[str] = []
    for sec in sections:
        kind = sec.kind.value
        if kind == "text":
            out.append(_escape_mdx_text(sec.value.strip()))
            out.append("")
        elif kind in ("parameters", "other parameters"):
            out.append("**Parameters:**")
            out.append("")
            for p in sec.value:
                anno = f" (`{p.annotation}`)" if p.annotation else ""
                desc = _escape_mdx_text(_one_line(p.description or ""))
                dash = f" — {desc}" if desc else ""
                out.append(f"- `{p.name}`{anno}{dash}")
            out.append("")
        elif kind == "returns":
            out.append("**Returns:**")
            out.append("")
            for r in sec.value:
                anno = f"`{r.annotation}` — " if r.annotation else ""
                desc = _escape_mdx_text(_one_line(r.description or ""))
                out.append(f"- {anno}{desc}")
            out.append("")
        elif kind in ("yields", "receives"):
            out.append(f"**{kind.capitalize()}:**")
            out.append("")
            for r in sec.value:
                anno = f"`{r.annotation}` — " if r.annotation else ""
                desc = _escape_mdx_text(_one_line(r.description or ""))
                out.append(f"- {anno}{desc}")
            out.append("")
        elif kind in ("raises", "warns"):
            label = "Raises" if kind == "raises" else "Warns"
            out.append(f"**{label}:**")
            out.append("")
            for r in sec.value:
                anno = f"`{r.annotation}` — " if r.annotation else ""
                desc = _escape_mdx_text(_one_line(r.description or ""))
                out.append(f"- {anno}{desc}")
            out.append("")
        elif kind == "examples":
            out.append("**Examples:**")
            out.append("")
            for example_kind, content in sec.value:
                # griffe yields (DocstringSectionKind, str) pairs; code parts
                # are tagged as "examples", prose as "text".
                if getattr(example_kind, "value", "") == "examples":
                    out.append("```python")
                    out.append(content.strip())
                    out.append("```")
                else:
                    out.append(_escape_mdx_text(content.strip()))
                out.append("")
        elif kind == "admonition":
            title = getattr(sec, "title", None) or ""
            if title:
                out.append(f"**{_escape_mdx_text(title)}:**")
                out.append("")
            val = sec.value
            text = val.contents if hasattr(val, "contents") else str(val)
            out.append(_escape_mdx_text(text.strip()))
            out.append("")
        else:
            # Unknown section: render its raw text if it's a string.
            val = sec.value
            if isinstance(val, str):
                out.append(_escape_mdx_text(val.strip()))
                out.append("")
    return "\n".join(out).rstrip() + "\n"


def _render_docstring(obj) -> str:
    """Return the parsed docstring as MDX-safe Markdown, or '' if none.

    Parses Google-style sections (Args, Returns, Raises, ...) so they render
    as formatted lists rather than a flat paragraph. Falls back to escaped
    raw text if parsing fails.
    """
    if not obj.docstring:
        return ""
    try:
        sections = obj.docstring.parse(griffe.Parser.google)
        return _render_docstring_sections(sections)
    except Exception:
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
    body_doc = ""
    if module.docstring:
        full = module.docstring.value.strip()
        # The first line becomes the page subtitle (frontmatter description).
        # Render only the remainder in the body to avoid showing the summary
        # line twice (once as subtitle, once as the first body paragraph).
        first_line, _, rest = full.partition("\n")
        desc = first_line.replace('"', "'")
        body_doc = rest.strip()

    lines = [
        "---",
        f'title: "{title}"',
    ]
    if desc:
        lines.append(f'description: "{desc}"')
    lines.append("---")
    lines.append("")

    if body_doc:
        lines.append(_escape_mdx_text(body_doc))
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
        # Render the top-level ``tsim`` package as the API landing page, and
        # any module that exposes public classes/functions. Skip sub-package
        # ``__init__`` modules that only re-export (no direct public members):
        # they produce a near-empty stub whose title duplicates both the nav
        # group and the same-named submodule (e.g. ``compile`` group +
        # ``compile.py`` page + ``compile/__init__`` page).
        is_root = path == "tsim"
        if has_public or is_root:
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
        # Pin matplotlib's embedded SVG date for reproducible output. Must be
        # set before the kernel (a subprocess) starts so it's inherited.
        old_sde = os.environ.get("SOURCE_DATE_EPOCH")
        os.environ["SOURCE_DATE_EPOCH"] = "1704067200"  # 2024-01-01 UTC
        # Inject a hidden preamble that seeds RNG and fixes matplotlib's SVG
        # element-id hashing, so executed outputs (sampled values, figures)
        # are byte-stable across builds and safe to commit. tsim samplers use
        # ``np.random.default_rng()`` when seed is None (not the legacy global
        # RNG), so we default that to a fixed seed. (NOTE: sinter spawns
        # worker subprocesses that this does not reach, so Monte-Carlo plots
        # produced via sinter may still vary slightly between builds.)
        preamble = nbformat.v4.new_code_cell(
            "import numpy as _np\n"
            "try:\n"
            "    import matplotlib as _mpl\n"
            "    _mpl.rcParams['svg.hashsalt'] = 'tsim-docs'\n"
            "except Exception:\n"
            "    pass\n"
            "_orig_default_rng = _np.random.default_rng\n"
            "def _seeded_default_rng(*a, **k):\n"
            "    if not a and 'seed' not in k:\n"
            "        return _orig_default_rng(0)\n"
            "    return _orig_default_rng(*a, **k)\n"
            "_np.random.default_rng = _seeded_default_rng\n"
            "_np.random.seed(0)\n"
        )
        preamble.metadata["tags"] = ["remove-cell"]
        nb.cells.insert(0, preamble)
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
            if old_sde is None:
                os.environ.pop("SOURCE_DATE_EPOCH", None)
            else:
                os.environ["SOURCE_DATE_EPOCH"] = old_sde

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

    # Replace tsim diagram HTML wrappers with iframe embeds backed by
    # standalone HTML files (preserves zoom/pan).
    url_prefix = f"/images/tutorials/{nb_path.stem}"
    body, extra_files = _extract_diagram_svgs(body, out_img_dir, url_prefix)
    for fpath, data in extra_files:
        fpath.write_bytes(data)

    # Same treatment for pyzx D3 diagrams (div placeholder + module script).
    body, pyzx_files = _extract_pyzx_iframes(body, out_img_dir, url_prefix)
    for fpath, data in pyzx_files:
        fpath.write_bytes(data)

    # Strip any orphaned <script> tags or empty graph placeholders that
    # didn't pair up above.
    body = _strip_script_and_pyzx(body)

    # Normalize block math so each $$ sits on its own line (Mintlify
    # requirement).
    body = _normalize_block_math(body)

    # Wrap nbconvert's 4-space indented cell outputs in fenced code blocks
    # so Mintlify renders them as code rather than collapsing whitespace.
    body = _fence_indented_outputs(body)

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
