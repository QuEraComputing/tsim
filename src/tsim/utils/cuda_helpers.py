"""CUDA-runtime helpers used by the sampler hot path.

Everything here degrades gracefully: the optional NVIDIA packages
(``cuda-bindings``, ``cupy``, ``cuquantum``, ``nvtx``) are imported lazily and
every helper has a pure-numpy fallback.

* :func:`copy_d2h` — device-to-host copy of a JAX array. With
  ``cuda-bindings`` it copies with ``cudaMemcpy`` into a *pooled* pinned
  host buffer; pinned destinations reach PCIe line rate, whereas the
  pageable path is bounded by a host-side staging copy. Pinning is
  expensive (``cudaHostAlloc`` costs ~0.1 ms per MB), so buffers are
  recycled through :class:`PinnedPool` instead of being allocated per call.
* :func:`nvtx_range` — NVTX annotation for Nsight Systems, no-op without
  the ``nvtx`` package.
* :func:`custabilizer_available` — whether cuStabilizer's sparse sampler
  can be used for on-device channel sampling.
"""

from __future__ import annotations

import contextlib
import ctypes
import threading
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:
    cudart: Any = None
    _CUDA_BINDINGS_AVAILABLE = False
else:
    try:
        from cuda.bindings import runtime as cudart

        _CUDA_BINDINGS_AVAILABLE = True
    except Exception:
        cudart = None
        _CUDA_BINDINGS_AVAILABLE = False

try:
    import nvtx as _nvtx
except Exception:  # pragma: no cover - optional dependency
    _nvtx = None


def cuda_bindings_available() -> bool:
    """Return True if ``cuda-bindings`` is importable."""
    return _CUDA_BINDINGS_AVAILABLE


@contextlib.contextmanager
def nvtx_range(message: str) -> Iterator[None]:
    """Annotate a code region for Nsight Systems (no-op without ``nvtx``)."""
    if _nvtx is None:
        yield
        return
    with _nvtx.annotate(message, domain="tsim"):
        yield


_custabilizer_state: bool | None = None


def custabilizer_available() -> bool:
    """Return True if cuStabilizer sampling can be used.

    Requires ``cupy`` and ``cuquantum.stabilizer`` to import and JAX's
    default backend to be a GPU (the samples are handed to JAX via DLPack
    without leaving the device). The result is cached.
    """
    global _custabilizer_state
    if _custabilizer_state is None:
        try:
            import importlib

            import jax

            importlib.import_module("cupy")
            importlib.import_module("cuquantum.stabilizer.dem_sampling")
            _custabilizer_state = bool(jax.default_backend() == "gpu")
        except Exception:
            _custabilizer_state = False
    return bool(_custabilizer_state)


def _src_on_host(src) -> bool:
    """Return True iff ``src`` lives on a CPU device.

    Its buffer pointer would be a host address and unsafe to feed to
    ``cudaMemcpy(..., DeviceToHost)``.
    """
    devices = getattr(src, "devices", None)
    if devices is None:
        return False
    try:
        ds = list(devices())
    except TypeError:
        ds = list(devices)
    return any(getattr(d, "platform", "").lower() == "cpu" for d in ds)


class _PinnedBlock:
    """A ``cudaHostAlloc``'d region; freed with ``cudaFreeHost`` on ``__del__``."""

    __slots__ = ("nbytes", "ptr")

    def __init__(self, nbytes: int):
        err, ptr = cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostAlloc({nbytes}) failed: {err}")
        self.ptr = int(ptr)
        self.nbytes = nbytes

    def __del__(self):
        if self.ptr:
            # cudart may be torn down at interpreter exit.
            with contextlib.suppress(Exception):
                cudart.cudaFreeHost(self.ptr)
            self.ptr = 0


class PinnedPool:
    """Recycles pinned host blocks handed out by :func:`alloc_pinned_numpy`.

    A block is returned to the pool when the last numpy view of it is
    garbage collected. ``acquire`` reuses the smallest free block that fits
    unless it is more than four times larger than the request, in which
    case a right-sized block is allocated instead. Two limits bound the
    amount of page-locked host memory:

    * ``max_pooled_bytes``: free blocks kept for reuse (largest evicted
      first).
    * ``max_outstanding_bytes``: pinned bytes alive in total (leased +
      pooled). When a fresh allocation would exceed it, ``acquire`` returns
      ``None`` and the caller falls back to pageable memory. This keeps
      callers that hold on to many results from pinning unbounded RAM.

    """

    def __init__(
        self,
        max_pooled_bytes: int = 2 * 1024**3,
        max_outstanding_bytes: int = 8 * 1024**3,
    ):
        """Create an empty pool with the given byte limits."""
        self.max_pooled_bytes = max_pooled_bytes
        self.max_outstanding_bytes = max_outstanding_bytes
        self._free: list[_PinnedBlock] = []
        self._outstanding = 0
        self._lock = threading.Lock()

    @property
    def pooled_bytes(self) -> int:
        """Total size of the free blocks currently held by the pool."""
        with self._lock:
            return sum(b.nbytes for b in self._free)

    @property
    def outstanding_bytes(self) -> int:
        """Total pinned bytes alive (leased to arrays plus pooled)."""
        with self._lock:
            return self._outstanding

    def acquire(self, nbytes: int) -> _PinnedBlock | None:
        """Return a block of at least ``nbytes`` bytes, or ``None`` if over budget."""
        with self._lock:
            fits = [b for b in self._free if nbytes <= b.nbytes <= 4 * nbytes]
            if fits:
                block = min(fits, key=lambda b: b.nbytes)
                self._free.remove(block)
                return block
            if self._outstanding + nbytes > self.max_outstanding_bytes:
                return None
            self._outstanding += nbytes
        try:
            return _PinnedBlock(nbytes)
        except RuntimeError:
            with self._lock:
                self._outstanding -= nbytes
            return None

    def release(self, block: _PinnedBlock) -> None:
        """Give ``block`` back; blocks beyond ``max_pooled_bytes`` are freed."""
        with self._lock:
            keep = block.nbytes <= self.max_pooled_bytes
            if keep:
                self._free.append(block)
                self._free.sort(key=lambda b: b.nbytes)
                total = sum(b.nbytes for b in self._free)
                while total > self.max_pooled_bytes and self._free:
                    evicted = self._free.pop()
                    total -= evicted.nbytes
                    self._outstanding -= evicted.nbytes
            else:
                self._outstanding -= block.nbytes
            # Blocks that are not kept are freed when their last reference drops.

    def clear(self) -> None:
        """Free all pooled blocks."""
        with self._lock:
            self._outstanding -= sum(b.nbytes for b in self._free)
            self._free.clear()


class _Lease:
    """Owner object stored on a numpy view; returns the block on collection."""

    __slots__ = ("block", "pool")

    def __init__(self, block: _PinnedBlock, pool: PinnedPool):
        self.block = block
        self.pool = pool

    def __del__(self):
        with contextlib.suppress(Exception):
            self.pool.release(self.block)


PINNED_POOL = PinnedPool()


def alloc_pinned_numpy(
    nbytes: int, dtype, shape, *, pool: PinnedPool | None = None
) -> np.ndarray:
    """Allocate a pinned host region from the pool and return it as an ndarray view.

    The returned array's ``base`` chain keeps a :class:`_Lease` alive; when
    the array and all derived views are dropped, the block returns to the
    pool (or is freed if the pool is full).

    Args:
        nbytes: Size of the underlying allocation in bytes. Must be at least
            ``prod(shape) * dtype.itemsize``.
        dtype: numpy-compatible dtype for the returned view.
        shape: Shape of the returned view.
        pool: Pool to draw from; defaults to the module-level pool.

    Returns:
        ndarray of the requested shape and dtype, backed by pinned memory.

    Raises:
        RuntimeError: if cuda.bindings is unavailable, the pool's pinned
            memory budget is exhausted, or ``cudaHostAlloc`` fails.

    """
    if not _CUDA_BINDINGS_AVAILABLE:
        raise RuntimeError(
            "cuda.bindings not importable; install 'cuda-bindings' or use "
            "copy_d2h() for a transparent fallback."
        )
    pool = PINNED_POOL if pool is None else pool
    block = pool.acquire(nbytes)
    if block is None:
        raise RuntimeError("pinned host memory budget exhausted")
    carr = (ctypes.c_uint8 * nbytes).from_address(block.ptr)
    carr._owner = _Lease(block, pool)  # type: ignore[attr-defined]  # arr.base = carr; carr._owner keeps the lease alive
    return np.frombuffer(carr, dtype=np.uint8).view(dtype).reshape(shape)


def empty_host(shape, dtype) -> np.ndarray:
    """Host array for receiving device data: pooled pinned if possible, else pageable."""
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    if _CUDA_BINDINGS_AVAILABLE and nbytes > 0:
        try:
            return alloc_pinned_numpy(nbytes, dtype, shape)
        except RuntimeError:
            pass
    return np.empty(shape, dtype=dtype)


def copy_d2h(src, *, dst: np.ndarray | None = None) -> np.ndarray:
    """Device-to-host copy, pinned-destination fast path when available.

    Args:
        src: Single-device contiguous array-like exposing
            ``unsafe_buffer_pointer()``, ``nbytes``, ``shape``, and ``dtype``.
            The caller must sync to the source's stream before invocation
            (``jax.block_until_ready(src)`` for a jax.Array).
        dst: Optional pre-allocated pinned ndarray to write into. Must have
            at least ``src.nbytes`` bytes. Defaults to a block from the
            pinned pool.

    Returns:
        ndarray with the same shape and dtype as ``src``.

    """
    if not _CUDA_BINDINGS_AVAILABLE or _src_on_host(src):
        # Fallback path. Two cases:
        #   1. cuda.bindings isn't importable
        #   2. src lives on a CPU device (JAX on a CPU jaxlib build, or any
        #      array marked CPU) — its ``unsafe_buffer_pointer`` is a host
        #      pointer, so ``cudaMemcpy(..., DeviceToHost)`` would fail.
        # ``np.asarray(jax.Array)`` returns a read-only zero-copy view in
        # newer JAX, so allocate fresh and copy in to a writable buffer.
        out = np.empty(src.shape, dtype=src.dtype)
        out[:] = src
        return out
    target: np.ndarray
    if dst is not None:
        target = dst
    else:
        try:
            target = alloc_pinned_numpy(src.nbytes, src.dtype, src.shape)
        except RuntimeError:
            # Pinned budget exhausted: fall back to a pageable destination
            # (cudaMemcpy stages through the driver's pinned buffers).
            target = np.empty(src.shape, dtype=src.dtype)
    err = cudart.cudaMemcpy(
        target.ctypes.data,
        src.unsafe_buffer_pointer(),
        src.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy d2h failed: {err}")
    return target
