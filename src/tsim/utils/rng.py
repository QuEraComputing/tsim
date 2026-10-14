"""SplitMix64 random numbers for numba kernels.

A kernel threads a uint64 state through these functions. Seeding each block of
work with its own 64-bit seed makes its numbers independent of how the blocks
are scheduled across threads.
"""

import numpy as np

from tsim.utils.kernels import jit

GOLDEN = np.uint64(0x9E3779B97F4A7C15)


@jit
def splitmix64(state: np.uint64) -> tuple[np.uint64, np.uint64]:
    """Advance a SplitMix64 state; return it and the next 64-bit output.

    ``splitmix64(seed + n * GOLDEN)`` is the stream's ``n + 1``-th output, so a
    kernel can also read the stream at any index.
    """
    state += GOLDEN
    z = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return state, z ^ (z >> np.uint64(31))


@jit
def next_uniform(state: np.uint64) -> tuple[np.uint64, float]:
    """Return the advanced state and a uniform float in [0, 1) with 53 bits."""
    state, z = splitmix64(state)
    return state, (z >> np.uint64(11)) * 2.0**-53


@jit
def next_skip(state: np.uint64, log_keep: float) -> tuple[np.uint64, float]:
    """Return the advanced state and the failures before the next success.

    The trials succeed with probability p and ``log_keep = log1p(-p)``. The
    uniform behind the draw has 64-bit resolution near 0, so the chance of no
    failure is exact for p down to about 2^-64. The count is a float: it is inf
    when the uniform rounds to 1, so callers compare it before converting it.
    """
    if log_keep == -np.inf:
        return state, 0.0
    state, z = splitmix64(state)
    return state, np.floor(np.log1p(-(z * 2.0**-64)) / log_keep)
