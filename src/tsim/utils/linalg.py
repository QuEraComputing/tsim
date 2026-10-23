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
    vecs = np.asarray(vectors, dtype=np.uint8)
    num_vectors = vecs.shape[0]

    # NOTE: vectors and their expansions over the basis are Python ints (bit j
    # is entry j). A vector is reduced by the basis vector whose lowest bit is
    # its own, until that bit is new (independent) or nothing is left.
    packed = np.packbits(vecs, axis=1, bitorder="little")
    reduced_basis: dict[int, tuple[int, int]] = {}
    basis_indices: list[int] = []
    t_rows: list[int] = []

    for i, row in enumerate(packed):
        v = int.from_bytes(row.tobytes(), "little")
        dep_sum = 0
        while v:
            pivot = (v & -v).bit_length() - 1
            if pivot not in reduced_basis:
                # reduced_v = v_original + sum(reducing vectors)
                # => reduced_v_expansion = e_new + sum(their expansions)
                rank = len(basis_indices)
                reduced_basis[pivot] = (v, dep_sum | 1 << rank)
                basis_indices.append(i)
                dep_sum = 1 << rank
                break
            b, expansion = reduced_basis[pivot]
            v ^= b
            dep_sum ^= expansion
        # A dependent vector is the sum of the expansions of its reducing vectors
        t_rows.append(dep_sum)

    rank = len(basis_indices)
    num_bytes = -(-rank // 8)
    t_bytes = b"".join(row.to_bytes(num_bytes, "little") for row in t_rows)
    transform = np.unpackbits(
        np.frombuffer(t_bytes, dtype=np.uint8).reshape(num_vectors, num_bytes),
        axis=1,
        count=rank,
        bitorder="little",
    )
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
