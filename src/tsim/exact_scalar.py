from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

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
    """Convert a (N, 4) array of coefficients to a (N,) array of numbers."""
    e4 = jnp.exp(1j * jnp.pi / 4)
    e4d = jnp.exp(-1j * jnp.pi / 4)
    return data[..., 0] + data[..., 1] * e4 + data[..., 2] * 1j + data[..., 3] * e4d


@jax.tree_util.register_pytree_node_class
class ExactScalarArray:
    def __init__(self, coeffs: jax.Array, power: Optional[jax.Array] = None):
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

    def tree_flatten(self):
        return ((self.coeffs, self.power), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

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

        self.coeffs, self.power = jax.lax.while_loop(
            cond_fun, body_fun, (self.coeffs, self.power)
        )
        return self

    def sum(self) -> "ExactScalarArray":
        """
        Sum elements along axis.
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
        # Move the reduction axis to position 0 for easier processing
        coeffs = jnp.moveaxis(self.coeffs, axis, 0)
        power = jnp.moveaxis(self.power, axis, 0)

        # Use associative scan to compute cumulative products, then take the last
        def scan_fn(a, b):
            return _scalar_mul(a, b)

        # lax.associative_scan computes cumulative products
        scanned = lax.associative_scan(scan_fn, coeffs, axis=0)

        # Take the last element along axis 0 (the final product)
        result_coeffs = scanned[-1]

        # Sum powers along the reduction axis
        result_power = jnp.sum(power, axis=0)

        return ExactScalarArray(result_coeffs, result_power)

    def to_complex(self) -> jax.Array:
        """Converts to complex number."""
        c_val = _scalar_to_complex(self.coeffs)
        scale = jnp.pow(2.0, self.power)
        return c_val * scale

    def to_numpy(self) -> np.ndarray:
        """Converts to complex128 numpy array."""
        numpy_data = np.array(self.coeffs)
        power = np.array(self.power)
        return 2.0**power * (
            numpy_data[..., 0]
            + numpy_data[..., 1] * np.exp(1j * np.pi / 4)
            + numpy_data[..., 2] * 1j
            + numpy_data[..., 3] * np.exp(-1j * np.pi / 4)
        )
