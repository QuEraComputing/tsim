"""Numba kernels that run in parallel only when the work pays for it."""

import threading
import types
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import numba

# NOTE: waking numba's threads costs about 0.1 ms, more than many kernel calls
# take serially. Numba's default threading layer aborts the process when two
# Python threads run parallel code at once, so parallel calls hold a lock; each
# one uses every core anyway.
PARALLEL_WORK = 1 << 16
_LOCK = threading.Lock()
_TASK_THREAD = threading.local()

_Task = TypeVar("_Task")
_Result = TypeVar("_Result")


def _renamed(function: Callable, suffix: str) -> Callable:
    """Return a copy of ``function`` whose name ends in ``suffix``."""
    renamed = types.FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__ + suffix,
        function.__defaults__,
        function.__closure__,
    )
    renamed.__qualname__ = function.__qualname__ + suffix
    renamed.__doc__ = function.__doc__
    return renamed


def jit(function: Callable, *, parallel: bool = False) -> Callable:
    """Compile ``function`` with numba, cached on disk where numba can write.

    The compiled function releases the GIL, so Python threads can run it at once.
    Numba refuses to define a cached function when no cache directory is
    writable (e.g. a read-only install without ``NUMBA_CACHE_DIR``), so such
    installs compile in every process instead.
    """
    # NOTE: numba names its cache files after the function and ignores nogil when
    # it reads them, so the name carries the flag: an entry cached without nogil
    # would keep holding the GIL.
    function = _renamed(function, "_nogil")
    try:
        return numba.njit(cache=True, parallel=parallel, nogil=True)(function)
    except RuntimeError:
        return numba.njit(parallel=parallel, nogil=True)(function)


def kernel(function: Callable) -> Callable:
    """Compile ``function`` serially and in parallel; ``run(work, *args)`` picks one.

    ``work`` estimates the call's cost in simple operations; calls below
    ``PARALLEL_WORK`` run serially, and so do calls on the worker threads of
    ``run_tasks``. In the serial compilation every ``numba.prange`` is a plain
    range, so both compute the same result.
    """
    serial = jit(function)
    # NOTE: numba names its cache files after the function, so the parallel
    # compilation needs another name to keep its own cache entry.
    parallel = jit(_renamed(function, "_parallel"), parallel=True)

    def run(work: int, *args):
        if work < PARALLEL_WORK or getattr(_TASK_THREAD, "active", False):
            return serial(*args)
        with _LOCK:
            return parallel(*args)

    run.__doc__ = function.__doc__
    return run


def run_tasks(
    function: Callable[[_Task], _Result], tasks: Iterable[_Task], threads: int
) -> list[_Result]:
    """Return ``[function(task) for task in tasks]``, on up to ``threads`` threads.

    With more than one thread the tasks run on worker threads, and every kernel
    in them runs serially: the tasks keep the cores busy, and independent tasks
    scale far better than numba's threads inside each kernel. Otherwise the
    tasks run in the calling thread.
    """
    tasks = list(tasks)
    threads = min(threads, len(tasks))
    if threads <= 1:
        return [function(task) for task in tasks]

    def run(task: _Task) -> _Result:
        _TASK_THREAD.active = True
        return function(task)

    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(run, tasks))
