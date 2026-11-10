import pytest

from tsim.statevector.cuda import runtime


@pytest.fixture(autouse=True)
def _cpu_samplers(request, monkeypatch):
    """Sample on the CPU, except in the CUDA tests, which choose per sampler."""
    if request.path.name != "test_cuda_sampler.py":
        monkeypatch.setattr(runtime, "available", lambda: False)
