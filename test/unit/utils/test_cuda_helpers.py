import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tsim.utils import cuda_helpers
from tsim.utils.cuda_helpers import (
    PinnedPool,
    alloc_pinned_numpy,
    copy_d2h,
    cuda_bindings_available,
    empty_host,
    nvtx_range,
)

needs_cuda = pytest.mark.skipif(
    not (cuda_bindings_available() and jax.default_backend() == "gpu"),
    reason="cuda-bindings and a GPU JAX backend are required",
)


def test_copy_d2h_roundtrip_any_backend():
    src = jnp.arange(12, dtype=jnp.int32).reshape(3, 4)
    out = copy_d2h(jax.block_until_ready(src))
    assert isinstance(out, np.ndarray)
    assert out.flags.writeable
    np.testing.assert_array_equal(out, np.arange(12).reshape(3, 4))


def test_nvtx_range_is_context_manager():
    with nvtx_range("tsim.test"):
        pass


@needs_cuda
def test_pool_reuses_blocks_after_collection():
    pool = PinnedPool()
    a = alloc_pinned_numpy(1 << 20, np.uint8, (1 << 20,), pool=pool)
    ptr = a.ctypes.data
    assert pool.outstanding_bytes == 1 << 20
    assert pool.pooled_bytes == 0
    del a
    gc.collect()
    assert pool.pooled_bytes == 1 << 20
    b = alloc_pinned_numpy(1 << 20, np.uint8, (1 << 20,), pool=pool)
    assert b.ctypes.data == ptr  # recycled
    assert pool.pooled_bytes == 0
    del b
    gc.collect()
    pool.clear()
    assert pool.outstanding_bytes == 0


@needs_cuda
def test_pool_budget_falls_back_to_pageable():
    pool = PinnedPool(max_outstanding_bytes=1 << 20)
    keep = alloc_pinned_numpy(1 << 19, np.uint8, (1 << 19,), pool=pool)
    with pytest.raises(RuntimeError, match="budget"):
        alloc_pinned_numpy(1 << 20, np.uint8, (1 << 20,), pool=pool)
    del keep
    gc.collect()
    # copy_d2h falls back to a pageable destination when the module pool is exhausted
    old = cuda_helpers.PINNED_POOL
    cuda_helpers.PINNED_POOL = PinnedPool(max_outstanding_bytes=0)
    try:
        src = jax.block_until_ready(jnp.ones((256,), dtype=jnp.float32))
        out = copy_d2h(src)
        np.testing.assert_array_equal(out, np.ones(256, dtype=np.float32))
        host = empty_host((4, 4), np.bool_)
        assert host.shape == (4, 4) and host.dtype == np.bool_
    finally:
        cuda_helpers.PINNED_POOL = old
