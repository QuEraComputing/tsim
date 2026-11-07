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


class SparseGF2Matrix:
    """A binary matrix stored as the sorted column indices of each row's ones.

    Used where a dense matrix would not fit: the error transform of a large
    circuit has ~10^5 rows and ~10^6 columns but only tens of ones per row.
    ``np.asarray`` gives the dense matrix.
    """

    def __init__(self, rows: list[tuple[int, ...]], num_cols: int):
        """Create the matrix from its rows' sorted column indices."""
        self.rows = rows
        self.num_cols = num_cols

    @classmethod
    def from_dense(cls, matrix: np.ndarray) -> "SparseGF2Matrix":
        """Return the sparse form of a 2-D matrix; nonzero entries count as 1."""
        dense = np.asarray(matrix)
        rows = [tuple(np.flatnonzero(row).tolist()) for row in dense]
        return cls(rows, dense.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """The shape ``(num_rows, num_cols)``."""
        return len(self.rows), self.num_cols

    def toarray(self) -> np.ndarray:
        """Return the dense uint8 matrix."""
        dense = np.zeros(self.shape, dtype=np.uint8)
        for i, row in enumerate(self.rows):
            dense[i, list(row)] = 1
        return dense

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Return the dense matrix, for ``np.asarray``."""
        if copy is False:
            raise ValueError("a SparseGF2Matrix can only be densified by copying")
        dense = self.toarray()
        return dense if dtype is None else dense.astype(dtype)

    def columns(self) -> list[tuple[int, ...]]:
        """Return each column's sorted row indices."""
        cols: list[list[int]] = [[] for _ in range(self.num_cols)]
        for i, row in enumerate(self.rows):
            for column in row:
                cols[column].append(i)
        return [tuple(c) for c in cols]


_EMPTY = (0, 0)


def _normalize(offset: int, bits: int) -> tuple[int, int]:
    """Shift ``(offset, bits)`` so that ``bits`` is odd, or return ``_EMPTY``."""
    if not bits:
        return _EMPTY
    shift = (bits & -bits).bit_length() - 1
    return offset + shift, bits >> shift


def _xor(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """XOR two normalized bit sets stored as ``(offset, bits)``, ``bits << offset``."""
    (offset_a, bits_a), (offset_b, bits_b) = a, b
    if not bits_a:
        return b
    if not bits_b:
        return a
    offset = min(offset_a, offset_b)
    return _normalize(
        offset, (bits_a << (offset_a - offset)) ^ (bits_b << (offset_b - offset))
    )


def _indices(offset: int, bits: int) -> list[int]:
    """Return the set bit positions of ``bits << offset``."""
    positions = []
    while bits:
        lowest = bits & -bits
        positions.append(offset + lowest.bit_length() - 1)
        bits ^= lowest
    return positions


def find_basis_sparse(
    rows: list[tuple[int, ...]],
) -> tuple[list[int], list[list[int]]]:
    """``find_basis`` for sparse rows, given as column indices (in any order).

    Each row is reduced by the basis vector whose lowest column is its own, until
    that column is new (the row joins the basis) or nothing is left. Each bit set
    is held as ``(offset, bits)`` with ``bits`` odd, so a Python int only spans a
    row's columns from its lowest set column, not from column 0. Rows whose
    columns are local (as an error-detecting circuit's are) stay small.

    Args:
        rows: Each row's column indices.

    Returns:
        The indices of the rows that form the basis, and for every row the
        sorted indices of the basis vectors whose XOR it is.

    """
    reduced_basis: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    basis_indices: list[int] = []
    expansions: list[list[int]] = []
    for i, columns in enumerate(rows):
        vector = _EMPTY
        if columns:
            lowest = min(columns)
            bits = 0
            for column in columns:
                bits ^= 1 << (column - lowest)
            vector = _normalize(lowest, bits)
        expansion = _EMPTY
        while vector[1]:
            pivot = vector[0]
            if pivot not in reduced_basis:
                # NOTE: the reduced row is the row plus the vectors that reduced
                # it, so its expansion is the new basis bit plus theirs.
                rank = len(basis_indices)
                reduced_basis[pivot] = (vector, _xor(expansion, (rank, 1)))
                basis_indices.append(i)
                expansion = (rank, 1)
                break
            basis_vector, basis_expansion = reduced_basis[pivot]
            vector = _xor(vector, basis_vector)
            expansion = _xor(expansion, basis_expansion)
        expansions.append(_indices(*expansion))
    return basis_indices, expansions


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
