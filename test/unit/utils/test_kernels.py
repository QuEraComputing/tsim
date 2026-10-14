import subprocess
import sys
import textwrap

import numba
import numpy as np

from tsim.utils.kernels import PARALLEL_WORK, kernel


@kernel
def _square_roots(out):
    for i in numba.prange(out.shape[0]):
        out[i] = np.sqrt(i)


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
