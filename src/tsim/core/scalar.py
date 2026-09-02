"""Abstract interface for batched scalar arithmetic used by the evaluator.

The compiled ZX scalar of every graph is a product of lookup-table entries
(``1 + ω^k``, ``ω^k``, ``±1``, ``1 + ω^a + ω^b - ω^(a+b)``) followed by a sum
over graphs. ``ScalarArray`` is the small algebra the term families and
``evaluate`` need to express that computation without committing to a
number representation:

* :class:`tsim.core.exact_scalar.ExactScalarArray` — exact dyadic
  arithmetic in the basis ``{1, ω, ω², ω³}`` with a separate power of two.
* :class:`tsim.core.float_scalar.Float32ScalarArray` /
  :class:`tsim.core.float_scalar.Float64ScalarArray` — a complex mantissa
  with a separate integer power of two (block floating point).

Every backend is an :class:`equinox.Module` so instances are pytrees and can
flow through ``jax.jit``. All lookup tables are expressed as exact dyadic
coefficient arrays of shape ``(..., 4)``; :meth:`ScalarArray.from_exact`
converts them into the backend's representation.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from typing_extensions import Self

Precision = Literal["exact", "float32", "float64"]
"""Number representation used for the per-graph scalar arithmetic.

* ``"exact"``: exact dyadic arithmetic (default). Products and sums over
  graphs are computed without rounding; only the final conversion to a
  complex amplitude rounds.
* ``"float32"``: complex64 mantissa with an int32 power of two.
* ``"float64"``: complex128 mantissa with an int32 power of two. Requires
  ``jax.config.update("jax_enable_x64", True)``.
"""

PRECISIONS: tuple[str, ...] = ("exact", "float32", "float64")


class ScalarArray(eqx.Module):
    """A batch of complex scalars with the operations ``evaluate`` needs.

    Shapes follow the leading (batch) dimensions of the backend's payload
    arrays; implementations may keep extra trailing dimensions (e.g. the
    four dyadic coefficients) that are not part of the logical shape.
    """

    @classmethod
    @abc.abstractmethod
    def from_exact(cls, coeffs: Array, power: Array | None = None) -> Self:
        """Build from exact dyadic coefficients ``(..., 4)`` and optional power of two.

        The value represented is
        ``(c_0 + c_1·ω + c_2·ω² + c_3·ω³) · 2^power`` with ``ω = e^{iπ/4}``.
        """

    @classmethod
    def one(cls) -> Self:
        """Return the multiplicative identity as a rank-0 scalar array."""
        return cls.from_exact(jnp.array([1, 0, 0, 0], dtype=jnp.int32))

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Logical shape (without representation-specific trailing axes)."""

    @classmethod
    @abc.abstractmethod
    def complex_dtype(cls) -> jax.typing.DTypeLike:
        """Return the dtype produced by :meth:`to_complex`."""

    @abc.abstractmethod
    def __mul__(self, other: Self) -> Self:
        """Element-wise product (with numpy broadcasting)."""

    @abc.abstractmethod
    def take(self, indices: Array) -> Self:
        """Gather along the first logical axis: ``result[i...] = self[indices[i...]]``."""

    @abc.abstractmethod
    def where(self, mask: Array, other: Self) -> Self:
        """Select ``self`` where ``mask`` is true, else ``other`` (broadcasting)."""

    @abc.abstractmethod
    def scale_power2(self, power: Array) -> Self:
        """Multiply by ``2**power`` element-wise (``power`` broadcasts)."""

    @abc.abstractmethod
    def prod(self, axis: int = -1) -> Self:
        """Product along ``axis`` (identity for an empty axis)."""

    @abc.abstractmethod
    def sum(self, axis: int = -1) -> Self:
        """Sum along ``axis``."""

    @abc.abstractmethod
    def to_complex(self) -> Array:
        """Convert to a complex array of the logical shape."""


def scalar_type(precision: str) -> type[ScalarArray]:
    """Return the :class:`ScalarArray` implementation for ``precision``.

    Raises:
        ValueError: for an unknown precision name.
        RuntimeError: for ``"float64"`` when JAX's 64-bit mode is disabled
            (JAX would silently truncate complex128 to complex64).

    """
    if precision == "exact":
        from tsim.core.exact_scalar import ExactScalarArray

        return ExactScalarArray
    if precision == "float32":
        from tsim.core.float_scalar import Float32ScalarArray

        return Float32ScalarArray
    if precision == "float64":
        if not bool(jax.config.read("jax_enable_x64")):
            raise RuntimeError(
                "precision='float64' requires JAX 64-bit mode. Enable it before "
                "creating arrays, e.g. `jax.config.update('jax_enable_x64', True)` "
                "or `JAX_ENABLE_X64=1`."
            )
        from tsim.core.float_scalar import Float64ScalarArray

        return Float64ScalarArray
    raise ValueError(f"Unknown precision {precision!r}; expected one of {PRECISIONS}")
