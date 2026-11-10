import pytest

from tsim.statevector.metal import runtime as metal_runtime


@pytest.fixture(autouse=True)
def _cpu_samplers(request, monkeypatch):
    """Sample on the CPU, except in the Metal tests, which choose per sampler."""
    if request.path.name != "test_metal_sampler.py":
        monkeypatch.setattr(metal_runtime, "available", lambda: False)
