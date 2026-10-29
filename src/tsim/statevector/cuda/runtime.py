"""CuPy and NVRTC for the CUDA executor, loaded on first use.

CuPy compiles the kernels at runtime with NVRTC. A pip-installed CUDA 13 NVRTC
(``nvidia-cuda-nvrtc``) is preloaded when present: CuPy may otherwise pick up
an older system NVRTC that cannot target the GPU (CUDA 12.0 cannot target
``sm_120``, e.g. the RTX 5090).
"""

from __future__ import annotations

import ctypes
import functools
import importlib.util
import pathlib
from types import ModuleType


@functools.cache
def cupy() -> ModuleType:
    """Return the ``cupy`` module, with a pip CUDA 13 NVRTC preloaded if installed.

    Raises:
        ImportError: If CuPy is not installed.

    """
    spec = importlib.util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations:
        for root in spec.submodule_search_locations:
            library = pathlib.Path(root) / "cu13" / "lib" / "libnvrtc.so.13"
            if library.exists():
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                break
    import cupy as cp

    return cp


def available() -> bool:
    """Return whether CuPy is installed and sees a CUDA device."""
    try:
        cp = cupy()
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@functools.cache
def arch() -> str:
    """Return the NVRTC architecture option of the current device, e.g. ``-arch=sm_120``."""
    return "-arch=sm_" + cupy().cuda.Device().compute_capability


@functools.cache
def multiprocessors() -> int:
    """Return the number of streaming multiprocessors of the current device."""
    return int(cupy().cuda.runtime.getDeviceProperties(0)["multiProcessorCount"])
