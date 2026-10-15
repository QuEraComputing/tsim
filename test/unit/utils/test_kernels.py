import subprocess
import sys
import textwrap
import threading

import numba
import numpy as np
import pytest

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


def test_run_tasks_raises_the_error_of_a_task():
    def task(n):
        if n == 2:
            raise ValueError("task 2")
        return n

    with pytest.raises(ValueError, match="task 2"):
        run_tasks(task, range(4), threads=3)


def test_the_calling_thread_keeps_parallel_kernels_after_run_tasks():
    if numba.get_num_threads() < 2:
        pytest.skip("needs two numba threads")
    run_tasks(lambda _: _thread_ids(PARALLEL_WORK, np.zeros(64)), range(4), 4)
    ids = np.full(1 << 12, -1)
    _thread_ids(PARALLEL_WORK, ids)
    assert len(np.unique(ids)) > 1
