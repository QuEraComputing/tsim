"""Exact scalar arithmetic for ZX-calculus phase computations.

Implements exact arithmetic for complex numbers of the form:
    (a + b*e^(i*pi/4) + c*i + d*e^(-i*pi/4)) * 2^power

This representation enables exact computation of phases in ZX-calculus graphs
without floating-point errors.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

_E4 = jnp.exp(1j * jnp.pi / 4)
_E4D = jnp.exp(-1j * jnp.pi / 4)


@jax.jit
def _scalar_mul(d1: jax.Array, d2: jax.Array) -> jax.Array:
    """Multiply two exact scalar coefficient arrays.

    Args:
        d1: Shape (..., 4) array of coefficients.
        d2: Shape (..., 4) array of coefficients.

    Returns:
        Shape (..., 4) array of product coefficients.

    """
    a1, b1, c1, d1_coeff = d1[..., 0], d1[..., 1], d1[..., 2], d1[..., 3]
    a2, b2, c2, d2_coeff = d2[..., 0], d2[..., 1], d2[..., 2], d2[..., 3]

    A = a1 * a2 + b1 * d2_coeff - c1 * c2 + d1_coeff * b2
    B = a1 * b2 + b1 * a2 + c1 * d2_coeff + d1_coeff * c2
    C = a1 * c2 + b1 * b2 + c1 * a2 - d1_coeff * d2_coeff
    D = a1 * d2_coeff - b1 * c2 - c1 * b2 + d1_coeff * a2

    return jnp.stack([A, B, C, D], axis=-1).astype(d1.dtype)


def _reduce_power_coeffs_step(
    power: jax.Array, coeffs: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Reduce one common factor of 2 from coefficients into the power."""
    reducible = jnp.all(coeffs % 2 == 0, axis=-1) & jnp.any(coeffs != 0, axis=-1)
    coeffs = jnp.where(reducible[..., None], coeffs // 2, coeffs)
    power = jnp.where(reducible, power + 1, power)
    return power, coeffs


def _scalar_mul_with_power(x: tuple, y: tuple) -> tuple:
    """Multiply two exact scalars represented as (power, coeffs) tuples.

    Delegates coefficient multiplication to `_scalar_mul` and applies a single
    reduction step (divides by 2) when all resulting coefficients are even.

    Args:
        x: Tuple of (power, coeffs) where coeffs has shape (..., 4).
        y: Tuple of (power, coeffs) where coeffs has shape (..., 4).

    Returns:
        Tuple of (new_power, new_coeffs).

    """
    p1, c1 = x
    p2, c2 = y

    new_coeffs = _scalar_mul(c1, c2)
    P = p1 + p2
    return _reduce_power_coeffs_step(P, new_coeffs)


def _scalar_add_with_power(x: tuple, y: tuple) -> tuple:
    """Add two exact scalars represented as (power, coeffs) tuples."""
    p1, c1 = x
    p2, c2 = y

    c1_scale = jnp.left_shift(jnp.ones_like(p1), jnp.maximum(p1 - p2, 0))[..., None]
    c2_scale = jnp.left_shift(jnp.ones_like(p2), jnp.maximum(p2 - p1, 0))[..., None]

    P = jnp.minimum(p1, p2)
    new_coeffs = c1 * c1_scale + c2 * c2_scale
    return _reduce_power_coeffs_step(P, new_coeffs)


def _scalar_to_complex(data: jax.Array) -> jax.Array:
    """Convert a (N, 4) array of coefficients to a (N,) array of complex numbers."""
    return data[..., 0] + data[..., 1] * _E4 + data[..., 2] * 1j + data[..., 3] * _E4D


class ExactScalarArray(eqx.Module):
    """Exact scalar array for ZX-calculus phase arithmetic using dyadic representation.

    Represents values of the form (c_0 + c_1·ω + c_2·ω² + c_3·ω³) × 2^power
    where ω = e^(iπ/4). This enables exact computation without floating-point errors.

    Attributes:
        coeffs: Array of shape (..., 4) containing dyadic coefficients.
        power: Array of powers of 2 for scaling.

    """

    coeffs: Array
    power: Array

    def __init__(self, coeffs: Array, power: Array | None = None):
        """Initialize from coefficients and optional power.

        The value represented is (c_0 + c_1*omega + c_2*omega^2 + c_3*omega^3) * 2^power
        where omega = e^{i*pi/4}.
        """
        self.coeffs = coeffs
        if power is None:
            self.power = jnp.zeros(coeffs.shape[:-1], dtype=jnp.int32)
        else:
            self.power = power

    def __mul__(self, other: "ExactScalarArray") -> "ExactScalarArray":
        """Element-wise multiplication."""
        new_coeffs = _scalar_mul(self.coeffs, other.coeffs)
        new_power = self.power + other.power
        return ExactScalarArray(new_coeffs, new_power)

    def sum(self, axis: int = -1) -> "ExactScalarArray":
        """Sum elements along the specified axis using normalized pairwise adds.

        Args:
            axis: The axis along which to sum.

        Returns:
            ExactScalarArray with the sum computed along the axis.

        """
        if axis < 0:
            axis += self.power.ndim

        scanned_power, scanned_coeffs = lax.associative_scan(
            _scalar_add_with_power, (self.power, self.coeffs), axis=axis
        )
        result_power = jnp.take(scanned_power, indices=-1, axis=axis)
        result_coeffs = jnp.take(scanned_coeffs, indices=-1, axis=axis)
        return ExactScalarArray(result_coeffs, result_power)

    def prod(self, axis: int = -1) -> "ExactScalarArray":
        """Compute product along the specified axis using associative scan.

        Returns identity (1+0i with power 0) for empty reductions.

        Args:
            axis: The axis along which to compute the product.

        Returns:
            ExactScalarArray with the product computed along the axis.

        """
        if axis < 0:
            axis += self.power.ndim

        if self.coeffs.shape[axis] == 0:
            # Product of empty sequence is identity: [1, 0, 0, 0] * 2^0
            coeffs_shape = self.coeffs.shape[:axis] + self.coeffs.shape[axis + 1 :]
            result_coeffs = jnp.zeros(coeffs_shape, dtype=self.coeffs.dtype)
            result_coeffs = result_coeffs.at[..., 0].set(1)

            power_shape = self.power.shape[:axis] + self.power.shape[axis + 1 :]
            result_power = jnp.zeros(power_shape, dtype=self.power.dtype)

            return ExactScalarArray(result_coeffs, result_power)

        scanned_power, scanned_coeffs = lax.associative_scan(
            _scalar_mul_with_power, (self.power, self.coeffs), axis=axis
        )
        result_power = jnp.take(scanned_power, indices=-1, axis=axis)
        result_coeffs = jnp.take(scanned_coeffs, indices=-1, axis=axis)

        return ExactScalarArray(result_coeffs, result_power)

    def to_complex(self) -> jax.Array:
        """Convert to complex number."""
        c_val = _scalar_to_complex(self.coeffs)
        scale = jnp.pow(2.0, self.power)
        return c_val * scale
