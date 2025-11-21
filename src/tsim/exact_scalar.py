from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

"""
This module implements exact scalar multiplication and segmentation for the exact scalar
arithmetic.

The exact scalar arithmetic is defined as the arithmetic of the complex numbers
of the form (a + b*e^(i*pi/4) + c*i + d*e^(-i*pi/4)) * 2^power

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


def _segment_mul_op(a, b):
    """Associative scan operator for segmented multiplication."""
    val_a, id_a = a
    val_b, id_b = b

    # If IDs match, multiply (accumulate).
    # If IDs differ, it means 'b' is the start of a new segment (or a jump),
    # so we just take 'val_b' as the new accumulator value.
    is_same = id_a == id_b

    prod = _scalar_mul(val_a, val_b)

    new_val = jnp.where(is_same[..., None], prod, val_b)

    # The ID always propagates from the right operand in the scan
    return new_val, id_b


@partial(jax.jit, static_argnames=["num_segments", "indices_are_sorted"])
def segment_scalar_prod(
    data: jax.Array,
    segment_ids: jax.Array,
    num_segments: int,
    indices_are_sorted: bool = False,
) -> jax.Array:
    """
    Compute the product of scalars within segments.

    Similar to jax.ops.segment_prod but for ExactScalar arithmetic.

    Args:
        data: Shape (..., 4) array of coefficients.
        segment_ids: Shape (...,) array of segment indices.
        num_segments: Total number of segments (determines output size).
        indices_are_sorted: If True, assumes segment_ids are sorted.

    Returns:
        Shape (..., num_segments, 4) array of products.
    """
    N = data.shape[0]
    if N == 0:
        return jnp.tile(jnp.array([1, 0, 0, 0], dtype=data.dtype), (num_segments, 1))

    if not indices_are_sorted:
        perm = jnp.argsort(segment_ids)
        data = data[perm]
        segment_ids = segment_ids[perm]

    # Associative scan to compute cumulative products within segments
    scanned_vals, _ = lax.associative_scan(_segment_mul_op, (data, segment_ids))

    # Identify the last element of each contiguous block of segment_ids
    # The last element holds the total product for that segment block.
    #
    # We must ensure that we only write once to each segment location to avoid
    # non-deterministic behavior on GPU (where scatter collisions are undefined).
    # Since segment_ids is sorted, we can identify the last occurrence of each ID.

    is_last = jnp.concatenate([segment_ids[:-1] != segment_ids[1:], jnp.array([True])])

    # Use a dummy index for non-last elements.
    # We extend res by 1 to have a trash bin at index 'num_segments'.
    dump_idx = num_segments
    scatter_indices = jnp.where(is_last, segment_ids, dump_idx)

    # Initialize result with multiplicative identity [1, 0, 0, 0]
    # Add one extra row for the dump
    res = jnp.tile(jnp.array([1, 0, 0, 0], dtype=data.dtype), (num_segments + 1, 1))

    # Scatter values. Only the last value of each segment is written to a valid index.
    # The rest go to the dump index.
    res = res.at[scatter_indices].set(scanned_vals)

    # Remove the dump row
    return res[:num_segments]


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

    def segment_prod(
        self, segment_ids: jax.Array, num_segments: int, indices_are_sorted: bool = True
    ) -> "ExactScalarArray":
        """
        Segmented product reduction.
        Generalizes segment_scalar_prod to ExactScalarArray.
        """
        return ExactScalarArray(
            segment_scalar_prod(
                self.coeffs, segment_ids, num_segments, indices_are_sorted
            ),
            jax.ops.segment_sum(
                self.power, segment_ids, num_segments, indices_are_sorted
            ),
        )

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
