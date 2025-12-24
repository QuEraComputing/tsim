from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

"""
This module implements exact scalar arithmetic for complex numbers of the form:
    (a + b*e^(i*pi/4) + c*i + d*e^(-i*pi/4)) * 2^power

This representation enables exact computation of phases in ZX-calculus graphs
without floating-point errors.

TODO: this representation can silently overflow. Add a check and raise an error.
"""


@jax.jit
def _scalar_mul(d1: jax.Array, d2: jax.Array) -> jax.Array:
    """
    Multiply two exact scalar coefficient arrays.

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


def _scalar_to_complex(data: jax.Array) -> jax.Array:
    """Convert a (N, 4) array of coefficients to a (N,) array of complex numbers."""
    e4 = jnp.exp(1j * jnp.pi / 4)
    e4d = jnp.exp(-1j * jnp.pi / 4)
    return data[..., 0] + data[..., 1] * e4 + data[..., 2] * 1j + data[..., 3] * e4d


class ExactScalarArray(eqx.Module):
    coeffs: Array
    power: Array

    def __init__(self, coeffs: Array, power: Optional[Array] = None):
        """
        Represents values of the form:
            (c_0 + c_1*omega + c_2*omega^2 + c_3*omega^3) * 2^power
        where omega = e^{i*pi/4}.
        """
        self.coeffs = coeffs
        if power is None:
            self.power = jnp.zeros(coeffs.shape[:-1], dtype=jnp.int32)
        else:
            self.power = power

    @classmethod
    def from_scalar_coeffs(cls, coeffs: jax.Array) -> "ExactScalarArray":
        """Creates ExactScalarArray with power=0"""
        return cls(coeffs)

    def __mul__(self, other: "ExactScalarArray") -> "ExactScalarArray":
        """Element-wise multiplication."""
        new_coeffs = _scalar_mul(self.coeffs, other.coeffs)
        new_power = self.power + other.power
        return ExactScalarArray(new_coeffs, new_power)

    def reduce(self) -> "ExactScalarArray":
        """
        Maximizes the power by dividing coefficients by 2 while they are all even.
        """

        def cond_fun(carry):
            coeffs, _ = carry
            # Reducible if all 4 components are even AND not all zero (0 is infinitely divisible)
            # We check 'not zero' to avoid infinite loops on strict 0.
            reducible = jnp.all(coeffs % 2 == 0, axis=-1) & jnp.any(
                coeffs != 0, axis=-1
            )
            return jnp.any(reducible)

        def body_fun(carry):
            coeffs, power = carry
            reducible = jnp.all(coeffs % 2 == 0, axis=-1) & jnp.any(
                coeffs != 0, axis=-1
            )
            coeffs = jnp.where(reducible[..., None], coeffs // 2, coeffs)
            power = jnp.where(reducible, power + 1, power)
            return coeffs, power

        new_coeffs, new_power = jax.lax.while_loop(
            cond_fun, body_fun, (self.coeffs, self.power)
        )
        return ExactScalarArray(new_coeffs, new_power)

    def sum(self) -> "ExactScalarArray":
        """
        Sum elements along the last axis (axis=-2).
        Aligns powers to the minimum power before summing.
        """
        # TODO: check for overflow and potentially refactor sum routine to scan
        # the array and reduce scalars every couple steps

        min_power = jnp.min(self.power, keepdims=False, axis=-1)
        pow = (self.power - min_power)[..., None]
        aligned_coeffs = self.coeffs * 2**pow
        summed_coeffs = jnp.sum(aligned_coeffs, axis=-2)
        return ExactScalarArray(summed_coeffs, min_power)

    def prod(self, axis: int = -1) -> "ExactScalarArray":
        """
        Compute product along the specified axis using associative scan.

        Args:
            axis: The axis along which to compute the product.

        Returns:
            ExactScalarArray with the product computed along the axis.
        """
        scanned = lax.associative_scan(_scalar_mul, self.coeffs, axis=axis)

        # Take the last element along the reduction axis
        result_coeffs = jnp.take(scanned, indices=-1, axis=axis)

        result_power = jnp.sum(self.power, axis=axis)

        return ExactScalarArray(result_coeffs, result_power)

    def to_complex(self) -> jax.Array:
        """Converts to complex number."""
        c_val = _scalar_to_complex(self.coeffs)
        scale = jnp.pow(2.0, self.power)
        return c_val * scale
