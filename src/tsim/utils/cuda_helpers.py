"""CUDA-runtime helpers used by the sampler hot path.

Lazy-imports ``cuda.bindings``. When the import succeeds, host-pinned
allocations + a direct ``cudaMemcpy`` replace numpy's default pageable d2h:
the pageable path forces the driver to stage the transfer through internal
pinned scratch before copying into the user buffer (host-DRAM-bandwidth
bound), while a pinned destination skips the staging hop and lets d2h reach
PCIe line rate. When the import fails, ``copy_d2h`` transparently falls
back to ``numpy.array``.
"""

from __future__ import annotations

import ctypes

import numpy as np

try:
    from cuda.bindings import runtime as cudart
    _CUDA_BINDINGS_AVAILABLE = True
except Exception:
    _CUDA_BINDINGS_AVAILABLE = False


class _PinnedBuf:
    """RAII wrapper for a ``cudaHostAlloc``'d region.

    The Python instance owns the lifetime; ``cudaFreeHost`` runs in
    ``__del__`` when no references remain (typically when the wrapping
    numpy view is garbage-collected).
    """

    __slots__ = ("ptr", "nbytes")

    def __init__(self, nbytes: int):
        err, ptr = cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostAlloc({nbytes}) failed: {err}")
        self.ptr = int(ptr)
        self.nbytes = nbytes

    def __del__(self):
        if self.ptr:
            try:
                cudart.cudaFreeHost(self.ptr)
            except Exception:
                # cudart may be torn down at interpreter exit.
                pass
            self.ptr = 0


def alloc_pinned_numpy(nbytes: int, dtype, shape) -> np.ndarray:
    """Allocate a pinned host region and return it as an ndarray view.

    The returned array's ``base`` chain pins the underlying ``_PinnedBuf``
    alive until the array and all derived views are dropped; only then does
    ``cudaFreeHost`` run.

    Args:
        nbytes: Size of the underlying allocation in bytes. Must be at least
            ``prod(shape) * dtype.itemsize``.
        dtype: numpy-compatible dtype for the returned view.
        shape: Shape of the returned view.

    Returns:
        ndarray of the requested shape and dtype, backed by pinned memory.

    Raises:
        RuntimeError: if cuda.bindings is unavailable, or the underlying
            ``cudaHostAlloc`` fails.
    """
    if not _CUDA_BINDINGS_AVAILABLE:
        raise RuntimeError(
            "cuda.bindings not importable; install 'cuda-bindings' or use "
            "copy_d2h() for a transparent fallback."
        )
    buf = _PinnedBuf(nbytes)
    carr = (ctypes.c_uint8 * nbytes).from_address(buf.ptr)
    carr._owner = buf  # arr.base = carr; carr._owner = buf → buf stays alive
    return np.frombuffer(carr, dtype=np.uint8).view(dtype).reshape(shape)


def copy_d2h(src, *, dst: np.ndarray | None = None) -> np.ndarray:
    """Device-to-host copy, pinned-destination fast path when available.

    Args:
        src: Single-device contiguous array-like exposing
            ``unsafe_buffer_pointer()``, ``nbytes``, ``shape``, and ``dtype``.
            The caller must sync to the source's stream before invocation
            (``jax.block_until_ready(src)`` for a jax.Array).
        dst: Optional pre-allocated pinned ndarray to write into. Must have
            at least ``src.nbytes`` bytes.

    Returns:
        ndarray with the same shape and dtype as ``src``.
    """
    if not _CUDA_BINDINGS_AVAILABLE:
        # np.asarray(jax.Array) is a read-only zero-copy view; callers
        # mutate the return (e.g. XOR detectors with the reference sample)
        # so allocate fresh and copy.
        out = np.empty(src.shape, dtype=src.dtype)
        out[:] = src
        return out
    if dst is None:
        dst = alloc_pinned_numpy(src.nbytes, src.dtype, src.shape)
    err = cudart.cudaMemcpy(
        dst.ctypes.data,
        src.unsafe_buffer_pointer(),
        src.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy d2h failed: {err}")
    return dst
