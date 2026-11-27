import pytest

from tsim.statevector.cuda import runtime as cuda_runtime
from tsim.statevector.metal import runtime as metal_runtime

GPU_TESTS = {
    "test_metal_sampler.py": metal_runtime,
    "test_cuda_sampler.py": cuda_runtime,
}


@pytest.fixture(autouse=True)
def _cpu_samplers(request, monkeypatch):
    """Sample on the CPU, except in each GPU's tests, which choose per sampler."""
    for name, runtime in GPU_TESTS.items():
        if request.path.name != name:
            monkeypatch.setattr(runtime, "unavailable", lambda: "a test on the CPU")
