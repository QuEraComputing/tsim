"""Tests for ``tsim.noise.fault_join_cuda``: the GPU join matches the numba one."""

import numpy as np
import pytest
import stim

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - no GPU
    pytest.skip("no CUDA device", allow_module_level=True)

from test.unit.noise.test_faults import GHZ, NOISES  # noqa: E402

from tsim import Circuit  # noqa: E402
from tsim.noise import faults  # noqa: E402
from tsim.noise.fault_join_cuda import CudaJoin  # noqa: E402


def _rows(a):
    return sorted(map(tuple, np.asarray(a).tolist()))


@pytest.mark.parametrize(
    "text",
    [
        GHZ,
        NOISES,
        str(
            stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=3,
                rounds=3,
                after_clifford_depolarization=0.01,
                before_measure_flip_probability=0.01,
            )
        ),
    ],
)
@pytest.mark.parametrize(
    ("weight", "small"), [(1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (5, 1)]
)
def test_the_gpu_join_finds_the_sets_of_the_cpu_join(text, weight, small):
    search = faults._Search(Circuit(text))
    join = faults._Join(search, weight, small=small)
    if join._large > 3:
        pytest.skip("the numba join streams at most 3 faults")
    gpu = CudaJoin(join)
    signs, sets = join.collect()
    gpu_signs, _ = gpu.collect()
    assert _rows(signs) == _rows(gpu_signs)
    if len(signs) == 0:
        return
    reps = np.bitwise_xor.reduce(search._bits[sets], axis=1)
    distributions = [b.outcomes(reps[:, b.f_selection]) for b in search._branches]
    table = search._table(signs, distributions)
    assert _rows(join.emit(table)) == _rows(gpu.emit(table))
