"""On-device channel sampling with NVIDIA cuStabilizer.

:class:`CuStabilizerChannelSampler` replaces the host-side geometric-skip
loop of :class:`~tsim.noise.channels.ChannelSampler` with cuStabilizer's
``BitMatrixSparseSampler``: independent Bernoulli errors are drawn into a
sparse device buffer and multiplied by the ``(n_errors, num_f)`` XOR-pattern
matrix over GF(2). tsim's categorical channels are first decomposed exactly
into independent Bernoullis (see
:meth:`ChannelSampler.independent_error_model`), so the sampled distribution
is identical to the numpy sampler's, not the ``approximate_disjoint_errors``
model used for detector error models.

Samples stay on the GPU: :meth:`sample_device` hands them to JAX through
DLPack, :meth:`sample_outputs` evaluates the direct-output bits with CuPy
and copies only those to the host through a pooled pinned buffer.

The module imports ``cupy`` and ``cuquantum`` lazily; use
:func:`tsim.utils.cuda_helpers.custabilizer_available` before constructing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tsim.utils.cuda_helpers import empty_host

if TYPE_CHECKING:
    import jax

# Device memory budget for one chunk of the direct path (bytes of the
# unpacked outcome matrix). Chunks keep the device footprint bounded for
# arbitrarily large ``shots``.
_DIRECT_CHUNK_BYTES = 256 * 1024**2


class CuStabilizerChannelSampler:
    """Samples f-variables on the GPU with cuStabilizer's sparse Bernoulli sampler.

    Args:
        probs: Shape ``(n_errors,)`` Bernoulli probabilities.
        patterns: Shape ``(n_errors, num_f)`` uint8 XOR patterns.
        seed: Seed of the host generator that draws a fresh device seed per
            ``sample`` call.
        device_id: CUDA device ordinal.

    """

    def __init__(
        self,
        probs: np.ndarray,
        patterns: np.ndarray,
        *,
        seed: int | None = None,
        device_id: int = 0,
    ):
        """Upload the error model; the device sampler itself is built lazily."""
        import cupy as cp

        if probs.ndim != 1 or patterns.shape != (probs.shape[0], patterns.shape[1]):
            raise ValueError("probs must be (n_errors,) and patterns (n_errors, num_f)")
        if probs.size == 0:
            raise ValueError("cuStabilizer sampling needs at least one error mechanism")
        self.num_errors = int(probs.shape[0])
        self.num_outputs = int(patterns.shape[1])
        self._device_id = device_id
        self._probs_d = cp.asarray(probs, dtype=cp.float64)
        self._patterns_d = cp.asarray(patterns, dtype=cp.uint8)
        self._rng = np.random.default_rng(seed)
        self._sampler: Any = None
        self._max_shots = 0

    def _ensure_capacity(self, num_shots: int) -> None:
        """(Re)build the device sampler if ``num_shots`` exceeds its capacity."""
        if self._sampler is not None and num_shots <= self._max_shots:
            return
        from cuquantum.stabilizer.dem_sampling import BitMatrixSparseSampler, Options

        # Grow geometrically so a slowly increasing batch size does not
        # trigger a rebuild every call.
        capacity = max(num_shots, 2 * self._max_shots)
        options = None if self._device_id == 0 else Options(device_id=self._device_id)
        self._sampler = BitMatrixSparseSampler(
            self._patterns_d,
            self._probs_d,
            max_shots=int(capacity),
            package="cupy",
            seed=int(self._rng.integers(0, 2**31 - 1)),
            options=options,
        )
        self._max_shots = int(capacity)

    def sample_cupy(self, num_shots: int):
        """Sample ``num_shots`` f-vectors; returns a C-contiguous CuPy uint8 array."""
        import cupy as cp

        if num_shots == 0:
            return cp.zeros((0, self.num_outputs), dtype=cp.uint8)
        self._ensure_capacity(num_shots)
        self._sampler.sample(int(num_shots), seed=int(self._rng.integers(0, 2**31 - 1)))
        out = self._sampler.get_outcomes(bit_packed=False)
        # cuStabilizer pads rows to a 32-bit boundary; JAX's DLPack import
        # requires compact strides, hence the copy.
        return cp.ascontiguousarray(out[:num_shots, : self.num_outputs])

    def sample_device(self, num_shots: int) -> jax.Array:
        """Sample ``num_shots`` f-vectors as a JAX ``uint8`` array on the GPU."""
        import cupy as cp
        import jax.numpy as jnp

        out = self.sample_cupy(num_shots)
        # CuPy and JAX use different streams; make the outcomes visible
        # before JAX reads the buffer.
        cp.cuda.get_current_stream().synchronize()
        return jnp.from_dlpack(out)

    def sample(self, num_shots: int) -> np.ndarray:
        """Sample ``num_shots`` f-vectors and return them on the host (uint8)."""
        import cupy as cp

        out = self.sample_cupy(num_shots)
        host = empty_host(out.shape, np.uint8)
        cp.asnumpy(out, out=host)
        return host

    def sample_outputs(
        self,
        num_shots: int,
        f_indices: np.ndarray,
        flips: np.ndarray,
        reindex: np.ndarray | None,
    ) -> np.ndarray:
        """Sample direct outputs ``f[:, f_indices] ^ flips`` (optionally reindexed).

        The selection runs on the device in chunks; only the ``(num_shots,
        len(f_indices))`` boolean result is copied to the host.
        """
        import cupy as cp

        n_out = len(f_indices)
        result = empty_host((num_shots, n_out), np.bool_)
        if num_shots == 0 or n_out == 0:
            return result
        idx_d = cp.asarray(np.asarray(f_indices, dtype=np.int64))
        flips_d = cp.asarray(np.asarray(flips, dtype=np.uint8))
        reindex_d = (
            None if reindex is None else cp.asarray(np.asarray(reindex, dtype=np.int64))
        )
        chunk = max(1, min(num_shots, _DIRECT_CHUNK_BYTES // max(self.num_outputs, 1)))
        for start in range(0, num_shots, chunk):
            stop = min(start + chunk, num_shots)
            f = self.sample_cupy(stop - start)
            bits = f[:, idx_d] ^ flips_d
            if reindex_d is not None:
                bits = bits[:, reindex_d]
            cp.asnumpy(
                cp.ascontiguousarray(bits), out=result[start:stop].view(np.uint8)
            )
        return result
