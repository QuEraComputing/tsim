import jax.numpy as jnp
import numpy as np
import pytest
from galois import GF2

from tsim.utils.linalg import (
    SparseGF2Matrix,
    find_basis,
    find_basis_sparse,
    matmul_gf2,
)


def verify_basis_decomposition(vectors: list[list[int]] | np.ndarray) -> None:
    vectors = np.array(vectors, dtype=np.uint8)
    basis, T = find_basis(vectors)

    assert T.shape == (len(vectors), len(basis))
    recon = (T @ basis) % 2
    assert np.array_equal(recon, vectors)

    r = np.linalg.matrix_rank(GF2(vectors))
    r_basis = np.linalg.matrix_rank(GF2(basis))
    r_combined = np.linalg.matrix_rank(GF2(np.vstack([vectors, basis])))

    assert len(basis) == r
    assert r_combined == r
    assert r_basis == r


def test_identity_basis():
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    verify_basis_decomposition(vectors)


def test_dependent_vectors():
    vectors = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],  # = v0 + v1
    ]
    verify_basis_decomposition(vectors)


def test_zero_vectors_and_duplicates():
    vectors = [
        [0, 0, 0],  # Zero vector
        [1, 0, 1],
        [1, 0, 1],  # Duplicate
        [0, 1, 0],
    ]
    verify_basis_decomposition(vectors)


def test_larger_mixed_case():
    vectors = [
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],  # v0 + v1
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    verify_basis_decomposition(vectors)


def test_empty_input():
    vectors = np.empty((0, 5), dtype=np.uint8)
    basis, T = find_basis(vectors)
    assert len(basis) == 0
    assert T.shape == (0, 0)


def test_single_vector():
    verify_basis_decomposition([[1, 0, 1]])
    verify_basis_decomposition([[0, 0, 0]])


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("shape", [(10, 10), (20, 5), (5, 20), (1, 50), (50, 1)])
def test_random_matrices(seed, shape):
    np.random.seed(seed)
    vectors = np.random.randint(0, 2, size=shape, dtype=np.uint8)
    verify_basis_decomposition(vectors)


def test_low_rank_random():
    np.random.seed(1)
    basis = np.random.randint(0, 2, size=(3, 10), dtype=np.uint8)
    # Create mixing matrix to generate 20 vectors from this basis
    mixing = np.random.randint(0, 2, size=(20, 3), dtype=np.uint8)

    vectors = (mixing @ basis) % 2
    verify_basis_decomposition(vectors)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("density", [0.02, 0.3, 0.8])
def test_basis_is_the_first_independent_vectors(seed, density):
    """The basis is every vector not spanned by the ones before it.

    Its vectors are independent, so they leave one transform, also pinned here.
    """
    rng = np.random.default_rng(seed)
    num_vectors, dim = int(rng.integers(1, 40)), int(rng.integers(1, 200))
    vectors = (rng.random((num_vectors, dim)) < density).astype(np.uint8)
    if num_vectors > 3:
        vectors[rng.integers(3, num_vectors)] = vectors[0] ^ vectors[1] ^ vectors[2]
    basis, transform = find_basis(vectors)

    first_independent, rank = [], 0
    for i in range(num_vectors):
        if (prefix_rank := np.linalg.matrix_rank(GF2(vectors[: i + 1]))) > rank:
            first_independent.append(i)
            rank = prefix_rank
    np.testing.assert_array_equal(basis, vectors[first_independent])
    assert transform.dtype == np.uint8 and transform.shape == (num_vectors, rank)
    np.testing.assert_array_equal(transform.astype(np.int64) @ basis % 2, vectors)


def test_matmul_gf2_matches_numpy_reference():
    np.random.seed(0)
    G, T, P, B = 3, 4, 17, 5
    a_np = np.random.randint(0, 2, size=(G, T, P), dtype=np.uint8)
    b_np = np.random.randint(0, 2, size=(B, P), dtype=np.uint8)

    expected = (b_np.astype(np.int64) @ a_np.astype(np.int64).reshape(G * T, P).T) % 2
    expected = expected.reshape(B, G, T).astype(np.uint8)

    result = np.array(matmul_gf2(jnp.asarray(a_np), jnp.asarray(b_np)))
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("p", [256, 300, 1024])
def test_matmul_gf2_no_uint8_saturation(p):
    """Regression: ``% 2`` must run on float32 before casting to uint8.

    JAX's float→uint8 cast saturates at 255 instead of wrapping mod 256, so
    casting before ``% 2`` silently corrupted parity once the inner-product
    summed more than 255 ones.
    """
    a = jnp.ones((1, 1, p), dtype=jnp.uint8)
    b = jnp.ones((1, p), dtype=jnp.uint8)
    result = np.array(matmul_gf2(a, b))
    assert result.flatten()[0] == p % 2


def test_matmul_gf2_empty_terms():
    a = jnp.zeros((0, 0, 4), dtype=jnp.uint8)
    b = jnp.zeros((2, 4), dtype=jnp.uint8)
    result = matmul_gf2(a, b)
    assert result.shape == (2, 0, 0)


def _reference_find_basis(vectors: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Eliminate dense rows by their lowest set bit, as Python ints."""
    reduced_basis: dict[int, tuple[int, int]] = {}
    basis_indices: list[int] = []
    expansions: list[int] = []
    for i, row in enumerate(vectors):
        vector = sum(1 << int(j) for j in np.flatnonzero(row))
        expansion = 0
        while vector:
            pivot = (vector & -vector).bit_length() - 1
            if pivot not in reduced_basis:
                rank = len(basis_indices)
                reduced_basis[pivot] = (vector, expansion | 1 << rank)
                basis_indices.append(i)
                expansion = 1 << rank
                break
            basis_vector, basis_expansion = reduced_basis[pivot]
            vector ^= basis_vector
            expansion ^= basis_expansion
        expansions.append(expansion)
    transform = np.array(
        [[(e >> k) & 1 for k in range(len(basis_indices))] for e in expansions],
        dtype=np.uint8,
    ).reshape(len(vectors), len(basis_indices))
    return basis_indices, transform


@pytest.mark.parametrize("seed", range(20))
def test_find_basis_matches_reference_elimination(seed):
    rng = np.random.default_rng(seed)
    num_rows, num_cols = rng.integers(1, 40), rng.integers(1, 200)
    dense = (rng.random((num_rows, num_cols)) < rng.uniform(0.01, 0.3)).astype(np.uint8)
    dense[rng.random(num_rows) < 0.2] = 0  # some empty rows
    if num_rows > 2:  # some dependent rows
        dense[-1] = dense[0] ^ dense[1]

    expected_indices, expected_transform = _reference_find_basis(dense)
    basis_indices, expansions = find_basis_sparse(
        SparseGF2Matrix.from_dense(dense).rows
    )
    assert basis_indices == expected_indices
    sparse_transform = np.zeros_like(expected_transform)
    for i, expansion in enumerate(expansions):
        sparse_transform[i, expansion] = 1
    np.testing.assert_array_equal(sparse_transform, expected_transform)

    basis, transform = find_basis(dense)
    np.testing.assert_array_equal(basis, dense[expected_indices])
    np.testing.assert_array_equal(transform, expected_transform)


def test_sparse_gf2_matrix_round_trip():
    rows = [(0, 3), (), (1, 2, 3)]
    matrix = SparseGF2Matrix(rows, 5)
    assert matrix.shape == (3, 5)
    dense = np.asarray(matrix)
    np.testing.assert_array_equal(
        dense, [[1, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 0]]
    )
    assert matrix.columns() == [(0,), (2,), (2,), (0, 2), ()]
    assert SparseGF2Matrix.from_dense(dense).rows == rows


def test_sparse_gf2_matrix_refuses_to_densify_without_copying():
    with pytest.raises(ValueError, match="copying"):
        np.asarray(SparseGF2Matrix([(0,)], 1), copy=False)
