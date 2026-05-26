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

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
API_DIR = DOCS_DIR / "api-reference"
TUTORIALS_DIR = DOCS_DIR / "tutorials"
IMAGES_TUT_DIR = DOCS_DIR / "images" / "tutorials"
DOCS_JSON = REPO_ROOT / "docs.json"
SRC_DIR = REPO_ROOT / "src"


def clean(api: bool, notebooks: bool) -> None:
    if api and API_DIR.exists():
        shutil.rmtree(API_DIR)
    if notebooks:
        if IMAGES_TUT_DIR.exists():
            shutil.rmtree(IMAGES_TUT_DIR)
        for mdx in TUTORIALS_DIR.glob("*.mdx"):
            mdx.unlink()


def build_api() -> int:
    print("[build_api] stub — not implemented yet")
    return 0


def build_notebooks(execute: bool) -> int:
    print(f"[build_notebooks] stub — not implemented yet (execute={execute})")
    return 0


def main() -> int:
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
