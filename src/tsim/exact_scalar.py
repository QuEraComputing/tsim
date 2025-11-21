from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

"""
This module implements exact scalar multiplication and segmentation for the exact scalar
arithmetic.

The exact scalar arithmetic is defined as the arithmetic of the complex numbers
of the form a + b*e^(i*pi/4) + c*i + d*e^(-i*pi/4)

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
class DyadicArray:
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
    def from_scalar_coeffs(cls, coeffs: jax.Array) -> "DyadicArray":
        """Creates DyadicArray with power=0"""
        return cls(coeffs)

    def tree_flatten(self):
        return ((self.coeffs, self.power), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __mul__(self, other: "DyadicArray") -> "DyadicArray":
        """Element-wise multiplication."""
        new_coeffs = _scalar_mul(self.coeffs, other.coeffs)
        new_power = self.power + other.power
        return DyadicArray(new_coeffs, new_power)

    def reduce(self) -> "DyadicArray":
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
        return DyadicArray(new_coeffs, new_power)

    def sum(self) -> "DyadicArray":
        """
        Sum elements along axis.
        Aligns powers to the minimum power before summing.
        """

        min_power = jnp.min(self.power, keepdims=False, axis=-1)
        pow = (self.power - min_power)[..., None]
        aligned_coeffs = self.coeffs * 2**pow
        summed_coeffs = jnp.sum(aligned_coeffs, axis=-2)
        return DyadicArray(summed_coeffs, min_power)

    def segment_prod(
        self, segment_ids: jax.Array, num_segments: int, indices_are_sorted: bool = True
    ) -> "DyadicArray":
        """
        Segmented product reduction.
        Generalizes segment_scalar_prod to DyadicArray.
        """
        if not indices_are_sorted:
            perm = jnp.argsort(segment_ids)
            coeffs = self.coeffs[perm]
            power = self.power[perm]
            segment_ids = segment_ids[perm]
        else:
            coeffs = self.coeffs
            power = self.power

        N = coeffs.shape[0]
        if N == 0:
            # Identity for prod is 1 * 2^0
            return DyadicArray(
                jnp.tile(
                    jnp.array([1, 0, 0, 0], dtype=coeffs.dtype), (num_segments, 1)
                ),
                jnp.zeros((num_segments,), dtype=power.dtype),
            )

        # Associative scan operator for (coeff, power) tuples
        def _dyadic_segment_mul_op(a, b):
            (c_a, p_a), id_a = a
            (c_b, p_b), id_b = b

            is_same = id_a == id_b

            # Product: multiply coeffs, add powers
            c_prod = _scalar_mul(c_a, c_b)
            p_prod = p_a + p_b

            new_c = jnp.where(is_same[..., None], c_prod, c_b)
            new_p = jnp.where(is_same, p_prod, p_b)

            return ((new_c, new_p), id_b)

        ((scanned_c, scanned_p), _) = lax.associative_scan(
            _dyadic_segment_mul_op, ((coeffs, power), segment_ids)
        )

        # Scatter to result
        is_last = jnp.concatenate(
            [segment_ids[:-1] != segment_ids[1:], jnp.array([True])]
        )
        dump_idx = num_segments
        scatter_indices = jnp.where(is_last, segment_ids, dump_idx)

        # Initialize result with identity
        res_c = jnp.tile(
            jnp.array([1, 0, 0, 0], dtype=coeffs.dtype), (num_segments + 1, 1)
        )
        res_p = jnp.zeros((num_segments + 1,), dtype=power.dtype)

        res_c = res_c.at[scatter_indices].set(scanned_c)
        res_p = res_p.at[scatter_indices].set(scanned_p)

        return DyadicArray(res_c[:num_segments], res_p[:num_segments])

    def to_complex(self) -> jax.Array:
        """Converts to complex number."""
        c_val = _scalar_to_complex(self.coeffs)
        scale = jnp.pow(2.0, self.power)
        return c_val * scale

    def to_numpy(self) -> np.ndarray:
        """Converts to complex numpy array."""
        numpy_data = np.array(self.coeffs, dtype=np.float64)
        power = np.array(self.power, dtype=np.float64)
        print("np data shape", numpy_data.shape)
        print("np power shape", power.shape)
        return 2.0**power * (
            numpy_data[..., 0]
            + numpy_data[..., 1] * np.exp(1j * np.pi / 4)
            + numpy_data[..., 2] * 1j
            + numpy_data[..., 3] * np.exp(-1j * np.pi / 4)
        )
