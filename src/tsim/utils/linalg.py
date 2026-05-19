"""Linear algebra utilities for GF(2) operations."""

import jax.numpy as jnp
import numpy as np
from jax import Array


def find_basis(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a set of binary vectors into a basis subset and a transformation matrix over GF(2).

    Given a set of vectors V, this function finds a maximal linearly independent subset B
    (the basis) and computes a transformation matrix T such that the original vectors can be
    reconstructed from the basis via matrix multiplication over GF(2):

    V = T @ B (mod 2)

    Args:
        vectors: Input binary vectors of shape `(N, D)`. Can be a list of lists or a numpy array.
                 Elements should be 0 or 1 (or convertible to them).

    Returns:
        A tuple `(basis, transform)` where:
            basis: The subset of independent vectors, shape `(K, D)`, where `K` is the rank.
            transform: The transformation matrix, shape `(N, K)`.

    """
    vecs = np.array(vectors, dtype=np.uint8)
    num_vectors, _ = vecs.shape

    basis_indices = []
    reduced_basis = []
    pivots = []
    basis_expansion = []
    t_rows = []

    for i in range(num_vectors):
        v = vecs[i].copy()
        coeffs = []

        for j, b in enumerate(reduced_basis):
            if v[pivots[j]]:
                v ^= b
                coeffs.append(j)

        is_independent = np.any(v)
        current_rank = len(basis_indices)
        new_size = current_rank + 1 if is_independent else current_rank

        # Compute dependency on existing basis vectors
        dep_sum = np.zeros(new_size, dtype=np.uint8)
        for idx in coeffs:
            e = basis_expansion[idx]
            dep_sum[: len(e)] ^= e

        if is_independent:
            basis_indices.append(i)
            reduced_basis.append(v)
            pivots.append(np.argmax(v))

            # Update basis expansion for the new reduced vector
            # reduced_v = v_original + sum(reduced_basis[c])
            # => reduced_v_expansion = e_new + sum(basis_expansion[c])
            dep_sum[current_rank] = 1
            basis_expansion.append(dep_sum)

            t_row = np.zeros(new_size, dtype=np.uint8)
            t_row[current_rank] = 1
            t_rows.append(t_row)
        else:
            # Dependent vector is the sum of basis expansions of reducing vectors
            t_rows.append(dep_sum)

    rank = len(basis_indices)
    transform = np.zeros((num_vectors, rank), dtype=np.uint8)
    for i, row in enumerate(t_rows):
        transform[i, : len(row)] = row

    return vecs[basis_indices], transform


def matmul_gf2(a: Array, b: Array) -> Array:
    """Compute binary dot products mod 2 as ``a_GTP x b_BP -> b_BGT``.

    Uses float32 matmul (integer matmul does not have BLAS support on CPU)
    then casts back to uint8.

    Args:
        a: Parameter bit-masks, shape ``(G, T, P)`` — G graphs, T terms, P parameters.
        b: Binary parameter values, shape ``(B, P)`` — B batch elements.

    Returns:
        Binary row-sums mod 2, shape ``(B, G, T)``.

    """
    G, T, _ = a.shape
    if G * T == 0:
        return jnp.zeros((b.shape[0], G, T), dtype=jnp.uint8)
    # NOTE: ``% 2`` must run on float32 — JAX's float→uint8 cast saturates at
    # 255 (it does not wrap mod 256), which would corrupt parity for inner
    # products with more than 255 set bits.
    sum_f32 = b.astype(jnp.float32) @ a.astype(jnp.float32).reshape(G * T, -1).T
    return (sum_f32.reshape(-1, G, T) % 2).astype(jnp.uint8)
