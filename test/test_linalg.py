import numpy as np
import pytest
from galois import GF2

from tsim.util.linalg import find_basis


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
