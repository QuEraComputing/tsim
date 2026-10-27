"""SplitMix64 random numbers for numba kernels.

A kernel threads a uint64 state through these functions. Seeding each block of
work with its own 64-bit seed makes its numbers independent of how the blocks
are scheduled across threads.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from numba import types
from numba.extending import intrinsic

from tsim.utils.kernels import jit

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_SQRT2 = 1.4142135623730951


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


def _bitcast(source, target) -> Callable:
    """Return a numba intrinsic that reinterprets a ``source`` value's bits as ``target``."""

    def typer(typingctx, x):
        def codegen(context, builder, signature, args):
            return builder.bitcast(args[0], context.get_value_type(target))

        return target(source), codegen

    return intrinsic(typer)


# NOTE: the casts only tell type checkers how the kernels call them.
_float_bits = cast(Callable[[float], np.uint64], _bitcast(types.float64, types.uint64))
_bits_float = cast(Callable[[np.uint64], float], _bitcast(types.uint64, types.float64))


@jit(contract=True)
def log1m(x: float) -> float:
    """Return log(1 - x) for x in [0, 1), within a few ulp of ``np.log1p(-x)``.

    It is inlined, unlike libm's log1p. With 1 - x = 2^e m, m in (sqrt(1/2),
    sqrt(2)], log(1 - x) = e log 2 + 2 atanh(s) with s = (m - 1)/(m + 1), summed
    to s^21 (|s| < 0.172). When e = 0, s = -x/(2 - x) is taken from x itself, so
    a small x keeps its precision. For x = 1 it returns about -709, not -inf.
    """
    u = 1.0 - x
    bits = _float_bits(u)
    e = np.int64((bits >> np.uint64(52)) & np.uint64(0x7FF)) - 1023
    m = _bits_float(
        (bits & np.uint64(0x000FFFFFFFFFFFFF)) | np.uint64(0x3FF0000000000000)
    )
    if m > _SQRT2:
        m *= 0.5
        e += 1
    # NOTE: both quotients, then a select: which one applies is random.
    near = -x / (2.0 - x)
    far = (m - 1.0) / (m + 1.0)
    s = near if e == 0 else far
    z = s * s
    p = 1.0 / 21.0
    p = p * z + 1.0 / 19.0
    p = p * z + 1.0 / 17.0
    p = p * z + 1.0 / 15.0
    p = p * z + 1.0 / 13.0
    p = p * z + 1.0 / 11.0
    p = p * z + 1.0 / 9.0
    p = p * z + 1.0 / 7.0
    p = p * z + 1.0 / 5.0
    p = p * z + 1.0 / 3.0
    p = p * z + 1.0
    return e * _LN2_HI + (2.0 * s * p + e * _LN2_LO)


@jit
def next_skip(state: np.uint64, log_keep: float) -> tuple[np.uint64, float]:
    """Return the advanced state and the failures before the next success.

    The trials succeed with probability p and ``log_keep = log1p(-p)``. The
    uniform behind the draw has 64-bit resolution near 0, so the chance of no
    failure is exact for p down to about 2^-64. The count is a float, so callers
    compare it before converting it; when the uniform rounds to 1 (chance 2^-54)
    it is ``709 / -log_keep`` rather than inf.
    """
    if log_keep == -np.inf:
        return state, 0.0
    state, z = splitmix64(state)
    return state, np.floor(log1m(z * 2.0**-64) / log_keep)
