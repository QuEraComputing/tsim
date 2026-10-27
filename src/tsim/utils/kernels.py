"""Numba kernels that run in parallel only when the work pays for it."""

import functools
import os
import threading
import types
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, wait
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


def _prange_only() -> dict[str, bool]:
    """Return numba's parallel options for explicit ``prange`` loops only.

    Numba would also run array expressions such as ``np.zeros`` or slice fills
    in parallel loops, each a thread launch, even inside a serial stretch.
    """
    options = ("comprehension", "reduction", "inplace_binop", "setitem", "numpy")
    return {**dict.fromkeys(options, False), "stencil": False, "fusion": False}


def jit(
    function: Callable | None = None, *, parallel: bool = False, contract: bool = False
) -> Callable:
    """Compile ``function`` with numba, cached on disk where numba can write.

    The compiled function releases the GIL, so Python threads can run it at once.
    Numba refuses to define a cached function when no cache directory is
    writable (e.g. a read-only install without ``NUMBA_CACHE_DIR``), so such
    installs compile in every process instead. With ``contract``, LLVM may fuse
    the function's multiplies and adds.
    """
    if function is None:
        return functools.partial(jit, parallel=parallel, contract=contract)
    # NOTE: numba names its cache files after the function and ignores nogil when
    # it reads them, so the name carries the flags: an entry cached without nogil
    # would keep holding the GIL.
    function = _renamed(function, "_nogil_contract" if contract else "_nogil")
    options = _prange_only() if parallel else False
    fastmath = {"contract"} if contract else False
    try:
        return numba.njit(cache=True, parallel=options, nogil=True, fastmath=fastmath)(
            function
        )
    except RuntimeError:
        return numba.njit(parallel=options, nogil=True, fastmath=fastmath)(function)


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
    parallel = jit(_renamed(function, "_prange"), parallel=True)

    def run(work: int, *args):
        if work < PARALLEL_WORK or getattr(_TASK_THREAD, "active", False):
            return serial(*args)
        with _LOCK:
            return parallel(*args)

    run.__doc__ = function.__doc__
    return run


def _mark_task_thread() -> None:
    _TASK_THREAD.active = True


@functools.cache
def _pool() -> ThreadPoolExecutor:
    # NOTE: one pool for every call, as large as numba's thread count can get.
    return ThreadPoolExecutor(
        numba.config.NUMBA_NUM_THREADS,  # pyright: ignore[reportAttributeAccessIssue]
        initializer=_mark_task_thread,
    )


def _after_fork_in_child() -> None:
    """Start over in a forked child: the parent's threads do not exist there."""
    global _LOCK
    _LOCK = threading.Lock()
    _pool.cache_clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def run_tasks(
    function: Callable[[_Task], _Result], tasks: Iterable[_Task], threads: int
) -> list[_Result]:
    """Return ``[function(task) for task in tasks]``, on up to ``threads`` threads.

    With more than one thread the tasks run on worker threads, and every kernel
    in them runs serially: the tasks keep the cores busy, and independent tasks
    scale far better than numba's threads inside each kernel. Otherwise, and in
    a task, the tasks run in the calling thread. Every task finishes before the
    first error, if any, is raised.
    """
    tasks = list(tasks)
    if min(threads, len(tasks)) <= 1 or getattr(_TASK_THREAD, "active", False):
        return [function(task) for task in tasks]
    slots = threading.Semaphore(threads)

    def run(task: _Task) -> _Result:
        with slots:
            return function(task)

    futures = [_pool().submit(run, task) for task in tasks]
    wait(futures)
    return [future.result() for future in futures]
