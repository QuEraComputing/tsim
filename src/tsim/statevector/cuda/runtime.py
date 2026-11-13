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
import os
import pathlib
from types import ModuleType

from tsim.statevector.gpu import FORKED


@functools.cache
def cupy() -> ModuleType:
    """Return the ``cupy`` module, with a pip CUDA 13 NVRTC preloaded if installed.

    Raises:
        ImportError: If CuPy is not installed.

    """
    if importlib.util.find_spec("cupy") is None:
        raise ImportError("CuPy is not installed")
    spec = importlib.util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations:
        for root in spec.submodule_search_locations:
            library = pathlib.Path(root) / "cu13" / "lib" / "libnvrtc.so.13"
            if library.exists():
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                break
    import cupy as cp

    return cp


_forked_after_init = False


def _after_fork_in_child() -> None:
    # NOTE: a child forked after its parent initialized CUDA cannot use CUDA; the
    # cached answer is read without touching CUDA.
    global _forked_after_init
    if _device_error.cache_info().currsize and _device_error() is None:
        _forked_after_init = True


def available() -> bool:
    """Return whether CuPy is installed and sees a CUDA device this process can use.

    ``CUDA_VISIBLE_DEVICES=""`` hides the devices, as in a process forked after
    its parent used the GPU.
    """
    return unavailable() is None


def unavailable() -> str | None:
    """Return why this process cannot sample on CUDA, or ``None`` if it can."""
    if _forked_after_init:
        return FORKED
    return _device_error()


def forked_after_init() -> bool:
    """Return whether this process was forked after its parent initialized CUDA."""
    return _forked_after_init


@functools.cache
def _device_error() -> str | None:
    try:
        cp = cupy()
        if cp.cuda.runtime.getDeviceCount() > 0:
            return None
        return "CuPy sees no CUDA device"
    except ImportError:
        return (
            "CuPy is not installed (install Tsim's cuda extra: "
            "pip install 'bloqade-tsim[cuda]')"
        )
    except Exception as error:
        return f"CuPy sees no usable CUDA device ({error})"


if hasattr(os, "register_at_fork"):  # not on Windows
    os.register_at_fork(after_in_child=_after_fork_in_child)


@functools.cache
def arch() -> str:
    """Return the NVRTC architecture option of the current device, e.g. ``-arch=sm_120``."""
    return "-arch=sm_" + cupy().cuda.Device().compute_capability


@functools.cache
def multiprocessors() -> int:
    """Return the number of streaming multiprocessors of the current device."""
    cuda = cupy().cuda
    return int(
        cuda.runtime.getDeviceProperties(cuda.Device().id)["multiProcessorCount"]
    )


def out_of_memory(error: BaseException) -> bool:
    """Return whether ``error`` is CuPy's or CUDA's out-of-memory error."""
    cuda = cupy().cuda
    if isinstance(error, cuda.memory.OutOfMemoryError):
        return True
    # NOTE: CUDA_ERROR_OUT_OF_MEMORY and cudaErrorMemoryAllocation are both 2.
    errors = (cuda.driver.CUDADriverError, cuda.runtime.CUDARuntimeError)
    return isinstance(error, errors) and getattr(error, "status", None) == 2


class CompileError(RuntimeError):
    """A CUDA kernel that NVRTC does not compile."""
