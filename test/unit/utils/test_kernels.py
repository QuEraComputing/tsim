import subprocess
import sys
import textwrap
import threading
import time

import numba
import numpy as np
import pytest

import tsim.utils.kernels as kernels
from tsim.utils.kernels import PARALLEL_WORK, kernel, run_tasks


@kernel
def _square_roots(out):
    for i in numba.prange(out.shape[0]):
        out[i] = np.sqrt(i)


@kernel
def _thread_ids(out):
    for i in numba.prange(out.shape[0]):
        out[i] = numba.get_thread_id()


def test_serial_and_parallel_runs_agree():
    serial, parallel = np.zeros(1000), np.zeros(1000)
    _square_roots(0, serial)
    _square_roots(PARALLEL_WORK, parallel)
    np.testing.assert_array_equal(serial, np.sqrt(np.arange(1000)))
    np.testing.assert_array_equal(parallel, serial)


def test_parallel_kernels_survive_concurrent_python_threads(tmp_path):
    # NOTE: without the lock numba's workqueue layer aborts the whole process, so
    # the threads run in a subprocess.
    script = tmp_path / "threads.py"
    script.write_text(textwrap.dedent("""
        import threading

        import numba
        import numpy as np

        from tsim.utils.kernels import kernel

        @kernel
        def fill(out):
            for i in numba.prange(out.shape[0]):
                out[i] = np.sqrt(i)

        def work():
            out = np.zeros(1 << 18)
            for _ in range(50):
                fill(1 << 30, out)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        """))
    command = [sys.executable, str(script)]
    result = subprocess.run(command, capture_output=True)  # noqa: S603
    assert result.returncode == 0, result.stderr.decode()


def test_run_tasks_keeps_the_order_and_runs_kernels_serially():
    def task(n):
        ids = np.full(1 << 12, -1)
        _thread_ids(PARALLEL_WORK, ids)
        return n * n, ids

    results = run_tasks(task, range(10), threads=4)
    assert [square for square, _ in results] == [n * n for n in range(10)]
    assert all((ids == 0).all() for _, ids in results)


def test_run_tasks_with_one_thread_runs_in_the_calling_thread():
    caller = threading.get_ident()
    assert run_tasks(lambda _: threading.get_ident(), range(3), 1) == [caller] * 3
    assert run_tasks(lambda n: n, [], 4) == []


def test_run_tasks_raises_the_error_of_a_task_once_every_task_is_done():
    finished = []

    def task(n):
        if n == 2:
            raise ValueError("task 2")
        time.sleep(0.05)
        finished.append(n)

    with pytest.raises(ValueError, match="task 2"):
        run_tasks(task, range(4), threads=3)
    assert sorted(finished) == [0, 1, 3]


def test_the_calling_thread_keeps_parallel_kernels_after_run_tasks():
    if numba.get_num_threads() < 2:
        pytest.skip("needs two numba threads")
    run_tasks(lambda _: _thread_ids(PARALLEL_WORK, np.zeros(64)), range(4), 4)
    ids = np.full(1 << 12, -1)
    _thread_ids(PARALLEL_WORK, ids)
    assert len(np.unique(ids)) > 1


def test_run_tasks_inside_a_task_runs_in_the_task():
    def task(n):
        return run_tasks(lambda m: (m, threading.get_ident()), range(3), 4)

    for inner in run_tasks(task, range(8), 4):
        assert [m for m, _ in inner] == [0, 1, 2]
        assert len({ident for _, ident in inner}) == 1


def test_run_tasks_runs_at_most_threads_tasks_at_once():
    lock, running, peak = threading.Lock(), [0], [0]

    def task(_):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.01)
        with lock:
            running[0] -= 1

    run_tasks(task, range(12), 3)
    assert 1 < peak[0] <= 3


def test_run_tasks_reuses_one_pool_for_every_thread_count():
    for threads in range(2, 9):
        run_tasks(lambda n: n, range(16), threads)
    assert kernels._pool.cache_info().currsize == 1
