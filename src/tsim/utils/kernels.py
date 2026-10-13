"""Numba kernels that run in parallel only when the work pays for it."""

import threading
import types
from collections.abc import Callable

import numba

# NOTE: waking numba's threads costs about 0.1 ms, more than many kernel calls
# take serially. Numba's default threading layer aborts the process when two
# Python threads run parallel code at once, so parallel calls hold a lock; each
# one uses every core anyway.
PARALLEL_WORK = 1 << 16
_LOCK = threading.Lock()


def jit(function: Callable, *, parallel: bool = False) -> Callable:
    """Compile ``function`` with numba, cached on disk where numba can write.

    Numba refuses to define a cached function when no cache directory is
    writable (e.g. a read-only install without ``NUMBA_CACHE_DIR``), so such
    installs compile in every process instead.
    """
    try:
        return numba.njit(cache=True, parallel=parallel)(function)
    except RuntimeError:
        return numba.njit(parallel=parallel)(function)


def kernel(function: Callable) -> Callable:
    """Compile ``function`` serially and in parallel; ``run(work, *args)`` picks one.

    ``work`` estimates the call's cost in simple operations; calls below
    ``PARALLEL_WORK`` run serially. In the serial compilation every
    ``numba.prange`` is a plain range, so both compute the same result.
    """
    serial = jit(function)
    # NOTE: numba names its cache files after the function, so the parallel
    # compilation needs another name to keep its own cache entry.
    renamed = types.FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__ + "_parallel",
        function.__defaults__,
        function.__closure__,
    )
    renamed.__qualname__ = function.__qualname__ + "_parallel"
    parallel = jit(renamed, parallel=True)

    def run(work: int, *args):
        if work < PARALLEL_WORK:
            return serial(*args)
        with _LOCK:
            return parallel(*args)

    run.__doc__ = function.__doc__
    return run
