"""The benchmark circuits: the tsim-bench repository at a pinned commit.

tsim-bench (https://github.com/rafaelha/tsim-bench) holds every benchmark circuit with the
paper it comes from, how that paper postselects, and its recorded active width per backend.
Its `tsim_bench` package (standard library only) reads that, and its worker times one
circuit. The pinned commit is exported once into the shared cache; set TSIM_BENCH_DIR, or
pass `--corpus-dir`, to use a local checkout instead (when changing tsim-bench itself).
"""

from __future__ import annotations

import fcntl
import functools
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import common
from .common import InfraError

URL = "https://github.com/rafaelha/tsim-bench"
# Bump together with any change that needs newer circuits or manifests.
REV = "0123c296702f7774b25172eac75c0bd4e0aca0a9"
COMPLETE = ".tsim-dev-complete"


def _export(rev: str) -> Path:
    """Export `rev` of tsim-bench into the cache once and return the directory.

    Runs that start together take a lock, and the export is extracted into a
    temporary directory and renamed into place, so a reader sees either no
    export or a complete one, and a finished export is never deleted.
    """
    cache = common.cache_dir()
    target = cache / "tsim-bench" / rev[:12]
    if (target / COMPLETE).exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(cache / "tsim-bench.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target / COMPLETE).exists():
            return target
        clone = cache / "tsim-bench.git"
        if not (clone / "HEAD").exists():
            print(f"fetching {URL} into {clone} ...", flush=True)
            shutil.rmtree(clone, ignore_errors=True)
            common.git("clone", "--bare", "-q", URL, str(clone))
        try:
            common.git("cat-file", "-e", f"{rev}^{{commit}}", cwd=clone)
        except InfraError:
            common.git("fetch", "-q", "origin", "+refs/heads/*:refs/heads/*", cwd=clone)
            try:
                common.git("cat-file", "-e", f"{rev}^{{commit}}", cwd=clone)
            except InfraError:
                raise InfraError(
                    f"tsim-bench has no commit {rev[:12]}: push it, or point "
                    "--corpus-dir / TSIM_BENCH_DIR at the checkout that has it"
                ) from None
        staging = Path(tempfile.mkdtemp(prefix=f".{rev[:12]}-", dir=target.parent))
        try:
            tarball = staging / "export.tar"
            common.git("archive", "-o", str(tarball), rev, cwd=clone)
            extract = subprocess.run(
                ["tar", "-x", "-f", str(tarball), "-C", str(staging)],
                capture_output=True,
                text=True,
                check=False,
            )
            if extract.returncode:
                raise InfraError(
                    f"extracting tsim-bench {rev[:12]}: {extract.stderr.strip()}"
                )
            tarball.unlink()
            (staging / COMPLETE).touch()
            # NOTE: a leftover from an interrupted run before this lock existed.
            shutil.rmtree(target, ignore_errors=True)
            staging.rename(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
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
