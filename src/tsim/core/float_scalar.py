"""Floating-point scalar arithmetic with a separate power of two.

``FloatScalarArray`` stores each scalar as ``mantissa · 2^power`` with a
complex ``mantissa`` and an ``int32`` ``power``. Products of many lookup-table
entries (``|1 + ω^k| ≤ 2``, ``|1 + ω^a + ω^b − ω^(a+b)| ≤ 2√2``) can span far
more than the exponent range of a single float, so long products are
evaluated in chunks whose partial results are renormalised with ``frexp``
before being combined. Sums align all summands to the largest power first.

The ``power`` field may be ``None``, meaning "all zero". Lookup-table entries
and their products carry no power until a chunked product or an explicit
:meth:`scale_power2` introduces one, which keeps the common short-product
path to a single fused ``prod``/``sum`` reduction.

Compared with :class:`tsim.core.exact_scalar.ExactScalarArray` this trades
exactness for 3–6× less memory traffic per scalar (one complex value instead
of four int32 coefficients plus an int32 power) and native complex
multiplies instead of 16 integer multiplies plus a parity check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from tsim.core.scalar import ScalarArray

if TYPE_CHECKING:
    from typing_extensions import Self

_OMEGA = np.exp(1j * np.pi / 4)


def _pow2(exponent: Array, real_dtype: jax.typing.DTypeLike) -> Array:
    """Exact ``2.0**exponent`` for integer ``exponent`` (``exp2`` is not exact)."""
    return jnp.ldexp(jnp.ones((), dtype=real_dtype), exponent)


def _add_power(a: Array | None, b: Array | None) -> Array | None:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


class FloatScalarArray(ScalarArray):
    """Block floating-point scalar array ``mantissa · 2^power``.

    Concrete dtypes are provided by :class:`Float32ScalarArray` and
    :class:`Float64ScalarArray`.

    Attributes:
        mantissa: Complex array holding the logical shape.
        power: ``int32`` array broadcastable to ``mantissa.shape``, or ``None``
            when every power is zero.

    """

    mantissa: Array
    power: Array | None

    _dtype: ClassVar[jax.typing.DTypeLike]
    _real_dtype: ClassVar[jax.typing.DTypeLike]
    # Largest number of lookup-table factors multiplied before renormalising.
    # Factor magnitudes lie in [2^-2, 2^1.5] (excluding exact zeros), so a
    # chunk stays inside the normal range of the dtype with a wide margin.
    _prod_chunk: ClassVar[int]

    def __init__(self, mantissa: Array, power: Array | None = None):
        """Wrap a complex ``mantissa`` and an optional integer ``power``."""
        self.mantissa = jnp.asarray(mantissa, dtype=self._dtype)
        self.power = power

    @classmethod
    def from_exact(cls, coeffs: Array, power: Array | None = None) -> Self:
        """Evaluate dyadic coefficients ``(..., 4)`` as a complex mantissa."""
        c = jnp.asarray(coeffs).astype(cls._real_dtype)
        w = jnp.asarray(_OMEGA, dtype=cls._dtype)
        wd = jnp.asarray(np.conj(_OMEGA), dtype=cls._dtype)
        mantissa = (
            c[..., 0]
            + c[..., 1] * w
            + c[..., 2] * jnp.asarray(1j, cls._dtype)
            + c[..., 3] * wd
        )
        return cls(mantissa, power)

    @classmethod
    def complex_dtype(cls) -> jax.typing.DTypeLike:
        """Return the mantissa dtype."""
        return cls._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical shape."""
        return self.mantissa.shape

    def __mul__(
        self, other: Self
    ) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Element-wise product; powers add, mantissas are not renormalised."""
        return type(self)(
            self.mantissa * other.mantissa, _add_power(self.power, other.power)
        )

    def take(self, indices: Array) -> Self:
        """Gather along the first logical axis."""
        power = None if self.power is None else self.power[indices]
        return type(self)(self.mantissa[indices], power)

    def where(
        self, mask: Array, other: Self
    ) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Select ``self`` where ``mask`` is true, else ``other``."""
        mantissa = jnp.where(mask, self.mantissa, other.mantissa)
        if self.power is None and other.power is None:
            power = None
        else:
            zero = jnp.zeros((), dtype=jnp.int32)
            power = jnp.where(
                mask,
                zero if self.power is None else self.power,
                zero if other.power is None else other.power,
            )
        return type(self)(mantissa, power)

    def scale_power2(self, power: Array) -> Self:
        """Multiply by ``2**power``."""
        return type(self)(self.mantissa, _add_power(self.power, power))

    def _normalize(self, mantissa: Array) -> tuple[Array, Array]:
        """Split ``mantissa`` into ``m · 2^e`` with ``|m| ∈ [0.5, 1)`` (or ``m = 0``)."""
        _, e = jnp.frexp(jnp.abs(mantissa))
        return mantissa * _pow2(-e, self._real_dtype), e.astype(jnp.int32)

    def prod(self, axis: int = -1) -> Self:
        """Product along ``axis``, renormalising every ``_prod_chunk`` factors."""
        m = self.mantissa
        if axis < 0:
            axis += m.ndim
        n = m.shape[axis]
        out_shape = m.shape[:axis] + m.shape[axis + 1 :]
        if n == 0:
            return type(self)(jnp.ones(out_shape, dtype=self._dtype), None)

        power = None
        if self.power is not None:
            power = jnp.sum(
                jnp.broadcast_to(self.power, m.shape), axis=axis, dtype=jnp.int32
            )

        chunk = self._prod_chunk
        if n <= chunk:
            return type(self)(jnp.prod(m, axis=axis), power)

        acc_m: Array | None = None
        acc_e: Array | None = None
        for start in range(0, n, chunk):
            part = lax.slice_in_dim(m, start, min(start + chunk, n), axis=axis)
            pm = jnp.prod(part, axis=axis)
            acc_m = pm if acc_m is None else acc_m * pm
            acc_m, e = self._normalize(acc_m)
            acc_e = e if acc_e is None else acc_e + e
        assert acc_m is not None
        return type(self)(acc_m, _add_power(power, acc_e))

    def sum(self, axis: int = -1) -> Self:
        """Sum along ``axis`` after aligning every summand to the largest power."""
        m = self.mantissa
        if axis < 0:
            axis += m.ndim
        if self.power is None:
            return type(self)(jnp.sum(m, axis=axis), None)
        p = jnp.broadcast_to(self.power, m.shape)
        pmax = jnp.max(p, axis=axis, keepdims=True)
        scaled = m * _pow2(p - pmax, self._real_dtype)
        return type(self)(jnp.sum(scaled, axis=axis), jnp.squeeze(pmax, axis=axis))

    def to_complex(self) -> Array:
        """Return ``mantissa · 2^power``."""
        if self.power is None:
            return self.mantissa
        return self.mantissa * _pow2(jnp.asarray(self.power), self._real_dtype)


class Float32ScalarArray(FloatScalarArray):
    """``complex64`` mantissa with ``int32`` power of two."""

    _dtype = jnp.complex64
    _real_dtype = jnp.float32
    _prod_chunk = 32  # 2^(1.5·32) = 2^48 ≪ 2^127


class Float64ScalarArray(FloatScalarArray):
    """``complex128`` mantissa with ``int32`` power of two (requires x64 mode)."""

    _dtype = jnp.complex128
    _real_dtype = jnp.float64
    _prod_chunk = 256  # 2^(1.5·256) = 2^384 ≪ 2^1023
