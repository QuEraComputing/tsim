from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

"""
This module implements exact scalar multiplication and segmentation for the exact scalar
arithmetic.

The exact scalar arithmetic is defined as the arithmetic of the complex numbers
of the form a + b*e^(i*pi/4) + c*i + d*e^(-i*pi/4)

TODO: this representation can easily overflow. Consider adding a 1/2**k prefix.
"""


@jax.jit
def scalar_mul(d1: jax.Array, d2: jax.Array) -> jax.Array:
    """
    Multiply two exact scalar coefficient arrays.

    Args:
        d1: Shape (..., 4) array of coefficients.
        d2: Shape (..., 4) array of coefficients.

    Returns:
        Shape (..., 4) array of product coefficients.
    """
    # Extract components. Works for (4,) or (..., 4)
    a1, b1, c1, d1_coeff = d1[..., 0], d1[..., 1], d1[..., 2], d1[..., 3]
    a2, b2, c2, d2_coeff = d2[..., 0], d2[..., 1], d2[..., 2], d2[..., 3]

    A = a1 * a2 + b1 * d2_coeff - c1 * c2 + d1_coeff * b2
    B = a1 * b2 + b1 * a2 + c1 * d2_coeff + d1_coeff * c2
    C = a1 * c2 + b1 * b2 + c1 * a2 - d1_coeff * d2_coeff
    D = a1 * d2_coeff - b1 * c2 - c1 * b2 + d1_coeff * a2

    return jnp.stack([A, B, C, D], axis=-1).astype(d1.dtype)


def _segment_mul_op(a, b):
    """Associative scan operator for segmented multiplication."""
    val_a, id_a = a
    val_b, id_b = b

    # If IDs match, multiply (accumulate).
    # If IDs differ, it means 'b' is the start of a new segment (or a jump),
    # so we just take 'val_b' as the new accumulator value.
    is_same = id_a == id_b

    prod = scalar_mul(val_a, val_b)

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
        data: Shape (N, 4) array of coefficients.
        segment_ids: Shape (N,) array of segment indices.
        num_segments: Total number of segments (determines output size).
        indices_are_sorted: If True, assumes segment_ids are sorted.

    Returns:
        Shape (num_segments, 4) array of products.
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


def scalar_to_complex(data: jax.Array) -> jax.Array:
    """Convert a (N, 4) array of coefficients to a (N,) array of numbers."""
    e4 = jnp.exp(1j * jnp.pi / 4)
    e4d = jnp.exp(-1j * jnp.pi / 4)
    return data[..., 0] + data[..., 1] * e4 + data[..., 2] * 1j + data[..., 3] * e4d
