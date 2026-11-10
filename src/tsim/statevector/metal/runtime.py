"""Metal through PyObjC for the Metal executor, loaded on first use.

Buffers are ``MTLStorageModeShared``: on Apple silicon the CPU and the GPU share
memory, so a buffer is also a NumPy array (``Buffer.array``) and nothing is
copied between them.
"""

from __future__ import annotations

import functools
import os
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np


@functools.cache
def metal() -> ModuleType:
    """Return PyObjC's ``Metal`` module.

    Raises:
        ImportError: If ``pyobjc-framework-Metal`` is not installed.

    """
    import Metal  # pyright: ignore[reportMissingImports]  # the metal extra, macOS only

    return Metal


@functools.cache
def device() -> Any:
    """Return the system's default Metal device.

    Raises:
        RuntimeError: If there is none.

    """
    global _initialized
    found = metal().MTLCreateSystemDefaultDevice()
    if found is None:
        raise RuntimeError("no Metal device")
    _initialized = True
    return found


_initialized = False
_forked_after_init = False


def _after_fork_in_child() -> None:
    # NOTE: Metal's device and its connections to the GPU driver do not survive a
    # fork; a child of a process that used them samples on the CPU.
    global _forked_after_init
    if _initialized:
        _forked_after_init = True


os.register_at_fork(after_in_child=_after_fork_in_child)


FORKED = (
    "This process was forked after its parent used the GPU, so it samples on the "
    "CPU; start it with the 'spawn' or 'forkserver' method to sample on the GPU."
)


def forked_after_init() -> bool:
    """Return whether this process was forked after its parent initialized Metal."""
    return _forked_after_init


@functools.cache
def queue() -> Any:
    """Return the executor's command queue."""
    return device().newCommandQueue()


@functools.cache
def check_simd_width() -> None:
    """Check that the device runs SIMD groups of 32 threads, as the kernels assume.

    Raises:
        RuntimeError: If it does not (e.g. an Intel or AMD GPU).

    """
    if not device().supportsFamily_(metal().MTLGPUFamilyApple7):
        raise RuntimeError("the Metal executor needs an Apple GPU (Apple silicon)")
    probe = Library(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void probe(device uint* out [[buffer(0)]]) { out[0] = 0; }"
    ).pipeline("probe")
    width = int(probe.threadExecutionWidth())
    if width != 32:
        raise RuntimeError(
            f"the Metal executor needs SIMD groups of 32 threads, not {width}"
        )


def available() -> bool:
    """Return whether PyObjC's Metal is installed and there is a usable Apple GPU.

    It is not in a process forked after its parent used the GPU.
    """
    return not _forked_after_init and _device_available()


@functools.cache
def _device_available() -> bool:
    try:
        check_simd_width()
        return True
    except Exception:
        return False


@dataclass
class Buffer:
    """A shared Metal buffer and a NumPy view of its memory."""

    handle: Any
    array: np.ndarray


POOL_BUFFERS = 2
"""Freed output buffers kept for reuse.

The GPU's first write to each page of a fresh buffer costs the OS a fault: for
rows of 440 bytes a call runs 1.7x longer than into a buffer that a previous
call returned (and its caller dropped).
"""

_pool: list[Any] = []
# NOTE: reentrant, since a buffer can be freed (``_Owner.__del__``) while this
# thread searches the pool.
_pool_lock = threading.RLock()


class _Owner:
    """Holds a Metal buffer for the NumPy arrays over its memory (their ``base``).

    A ``recycle`` buffer goes back to the pool when they are gone.
    """

    def __init__(self, handle: Any, nbytes: int, recycle: bool = False) -> None:
        self.handle = handle
        self.recycle = recycle
        view = handle.contents().as_buffer(nbytes)
        address = np.frombuffer(view, dtype=np.uint8).ctypes.data
        self.__array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (address, False),
            "version": 3,
        }

    def __del__(self) -> None:
        if self.recycle and _pool is not None:
            with _pool_lock:
                _pool.append(self.handle)
                del _pool[:-POOL_BUFFERS]


def buffer(count: int, dtype, recycle: bool = False) -> Buffer:
    """Return a shared buffer of ``count`` elements of ``dtype`` (zeroed).

    Its array keeps the Metal buffer alive, so it may outlive the ``Buffer``.
    With ``recycle`` the buffer may be a freed one of up to twice the size, not
    zeroed, and returns to the pool once its arrays are gone.
    """
    dtype = np.dtype(dtype)
    nbytes = max(int(count) * dtype.itemsize, 16)
    handle = None
    if recycle:
        with _pool_lock:
            fits = [b for b in _pool if nbytes <= b.length() <= 2 * nbytes]
            # NOTE: by identity, in case a buffer freed meanwhile trimmed the pool.
            if fits and any(b is fits[0] for b in _pool):
                handle = fits[0]
                _pool[:] = [b for b in _pool if b is not handle]
    if handle is None:
        handle = device().newBufferWithLength_options_(
            nbytes, metal().MTLResourceStorageModeShared
        )
    if handle is None:
        raise MemoryError(
            f"no Metal buffer of {nbytes} bytes (at most "
            f"{device().maxBufferLength()}); sample fewer shots per call"
        )
    owner = _Owner(handle, nbytes, recycle)
    array = np.asarray(owner)[: count * dtype.itemsize].view(dtype)
    return Buffer(handle, array)


def upload(values: np.ndarray) -> Buffer:
    """Return a shared buffer holding a copy of ``values`` (flattened)."""
    values = np.ascontiguousarray(values)
    held = buffer(values.size, values.dtype)
    held.array[:] = values.reshape(-1)
    return held


class Library:
    """A compiled Metal library and its kernels' pipeline states."""

    def __init__(self, source: str, fast_math: bool = True) -> None:
        """Compile ``source`` (Metal Shading Language).

        Raises:
            RuntimeError: If it does not compile.

        """
        mtl = metal()
        options = mtl.MTLCompileOptions.new()
        options.setFastMathEnabled_(fast_math)
        library, error = device().newLibraryWithSource_options_error_(
            source, options, None
        )
        if library is None:
            raise RuntimeError(f"Metal compilation failed: {error}")
        self._library = library
        self._pipelines: dict[str, Any] = {}

    def pipeline(self, name: str) -> Any:
        """Return the compute pipeline state of kernel ``name``."""
        if name not in self._pipelines:
            function = self._library.newFunctionWithName_(name)
            if function is None:
                raise KeyError(name)
            descriptor = metal().MTLComputePipelineDescriptor.new()
            descriptor.setComputeFunction_(function)
            descriptor.setLabel_(name)
            (
                state,
                error,
            ) = device().newComputePipelineStateWithDescriptor_options_reflection_error_(
                descriptor, 0, None, None
            )
            if state is None:
                raise RuntimeError(f"pipeline {name}: {error}")
            self._pipelines[name] = state
        return self._pipelines[name]


Argument = Buffer | tuple[Buffer, int] | np.ndarray | np.generic

PROFILE: dict[str, float] | None = None
"""If a dict, every dispatch runs as its own command buffer and adds its GPU time
(seconds) under its kernel's name: for profiling only."""


class Encoder:
    """Records kernel dispatches into one command buffer."""

    def __init__(self) -> None:
        """Start a command buffer with a compute encoder."""
        self.commands = queue().commandBuffer()
        self._encoder = self.commands.computeCommandEncoder()

    def dispatch(
        self,
        pipeline: Any,
        arguments: list[Argument],
        threadgroups: int,
        threads: int,
        threadgroup_bytes: int = 0,
    ) -> None:
        """Dispatch ``threadgroups`` groups of ``threads`` threads of ``pipeline``.

        Each argument binds buffer index ``i``: a buffer (or a buffer and a byte
        offset), or a NumPy scalar or array passed by value. A nonzero
        ``threadgroup_bytes`` binds threadgroup memory 0.
        """
        mtl, encoder = metal(), self._encoder
        encoder.setComputePipelineState_(pipeline)
        for i, argument in enumerate(arguments):
            if isinstance(argument, Buffer):
                encoder.setBuffer_offset_atIndex_(argument.handle, 0, i)
            elif isinstance(argument, tuple):
                encoder.setBuffer_offset_atIndex_(argument[0].handle, argument[1], i)
            else:
                data = np.ascontiguousarray(argument).tobytes()
                encoder.setBytes_length_atIndex_(data, len(data), i)
        if threadgroup_bytes:
            encoder.setThreadgroupMemoryLength_atIndex_(
                -(-threadgroup_bytes // 16) * 16, 0
            )
        if threadgroups:
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                mtl.MTLSizeMake(threadgroups, 1, 1), mtl.MTLSizeMake(threads, 1, 1)
            )
        if PROFILE is not None:
            name = str(pipeline.label() or "") or str(pipeline)
            commands = self.commit()
            PROFILE[name] = PROFILE.get(name, 0.0) + wait(commands)
            self.commands = queue().commandBuffer()
            self._encoder = self.commands.computeCommandEncoder()

    def commit(self) -> Any:
        """End encoding and commit; return the command buffer."""
        self._encoder.endEncoding()
        self.commands.commit()
        return self.commands


def wait(commands: Any) -> float:
    """Wait for a committed command buffer; return its GPU time in seconds.

    Raises:
        MemoryError: If the GPU ran out of memory.
        RuntimeError: If it failed otherwise.

    """
    commands.waitUntilCompleted()
    error = commands.error()
    if error is not None:
        if error.code() == metal().MTLCommandBufferErrorOutOfMemory:
            raise MemoryError(f"the GPU ran out of memory: {error}")
        raise RuntimeError(f"Metal command buffer failed: {error}")
    return float(commands.GPUEndTime() - commands.GPUStartTime())
