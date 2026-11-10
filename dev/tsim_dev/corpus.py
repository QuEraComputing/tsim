"""The benchmark circuits: the tsim-bench repository at a pinned commit.

tsim-bench (https://github.com/rafaelha/tsim-bench) holds every benchmark circuit with the
paper it comes from, how that paper postselects, and its recorded active width per backend.
Its `tsim_bench` package (standard library only) reads that, and its worker times one
circuit. The pinned commit is exported once into the shared cache; set TSIM_BENCH_DIR, or
pass `--corpus-dir`, to use a local checkout instead (when changing tsim-bench itself).
"""

from __future__ import annotations

import functools
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import common
from .common import InfraError

URL = "https://github.com/rafaelha/tsim-bench"
# Bump together with any change that needs newer circuits or manifests.
REV = "4cb8ca2963b2b982a4e37372b7168b4f590c4e97"


def _export(rev: str) -> Path:
    """Export `rev` of tsim-bench into the cache once and return the directory."""
    target = common.cache_dir() / "tsim-bench" / rev[:12]
    if (target / "tsim_bench").is_dir():
        return target
    clone = common.cache_dir() / "tsim-bench.git"
    if not clone.is_dir():
        print(f"fetching {URL} into {clone} ...", flush=True)
        common.git("clone", "--bare", "-q", URL, str(clone))
    try:
        common.git("cat-file", "-e", f"{rev}^{{commit}}", cwd=clone)
    except InfraError:
        common.git("fetch", "-q", "origin", "+refs/heads/*:refs/heads/*", cwd=clone)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    archive = subprocess.Popen(
        ["git", "archive", rev], cwd=clone, stdout=subprocess.PIPE
    )
    subprocess.run(["tar", "-x", "-C", str(target)], stdin=archive.stdout, check=True)
    if archive.wait():
        shutil.rmtree(target, ignore_errors=True)
        raise InfraError(f"git archive {rev} of tsim-bench failed")
    return target


@functools.cache
def root(explicit: str | None = None) -> Path:
    """Return the tsim-bench checkout to read circuits from, fetching it if needed."""
    for candidate in (explicit, os.environ.get("TSIM_BENCH_DIR")):
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if not (path / "tsim_bench").is_dir():
                raise InfraError(
                    f"{path} is not a tsim-bench checkout (no tsim_bench/)"
                )
            return path
    return _export(REV)


@functools.cache
def package(explicit: str | None = None):
    """Import `tsim_bench` from the checkout and return the module."""
    path = str(root(explicit))
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("tsim_bench")


def worker(explicit: str | None = None) -> Path:
    """Return the path of the timing worker of the checkout."""
    return root(explicit) / "tsim_bench" / "worker.py"


def describe(explicit: str | None = None) -> str:
    """Name the tsim-bench in use: the pinned commit, or a local checkout's path."""
    if explicit or os.environ.get("TSIM_BENCH_DIR"):
        path = root(explicit)
        return f"{common.describe(path)} ({path})"
    return REV[:8]
