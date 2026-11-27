"""Tests for ``tsim.utils.gf2_span``."""

import numpy as np
import pytest

import tsim.utils.gf2_span
from tsim.utils.gf2_span import Span, pack, pack_bytes, unpack


@pytest.fixture(autouse=True, params=["numba", "python"])
def _kernels(request, monkeypatch):
    # NOTE: coverage cannot see inside numba kernels, so run them as Python too.
    if request.param == "python":
        for name in ("_lowest", "_has_bit", "_insert", "_annihilator"):
            kernel = getattr(tsim.utils.gf2_span, name)
            monkeypatch.setattr(tsim.utils.gf2_span, name, kernel.py_func)


def _rank(matrix: np.ndarray) -> int:
    matrix = matrix.copy() % 2
    rank = 0
    for col in range(matrix.shape[1]):
        rows = np.flatnonzero(matrix[rank:, col]) + rank
        if not len(rows):
            continue
        matrix[[rank, rows[0]]] = matrix[[rows[0], rank]]
        for r in np.flatnonzero(matrix[:, col]):
            if r != rank:
                matrix[r] ^= matrix[rank]
        rank += 1
        if rank == len(matrix):
            break
    return rank


@pytest.mark.parametrize("width", [0, 1, 7, 64, 65, 130])
def test_pack_round_trips(width):
    bits = np.random.default_rng(width).integers(0, 2, (5, width), dtype=np.uint8)
    assert np.array_equal(unpack(pack(bits), width), bits)
    packed = np.packbits(bits, axis=1, bitorder="little")
    assert np.array_equal(pack_bytes(packed, width), pack(bits))


@pytest.mark.parametrize("seed", range(6))
def test_the_annihilator_is_the_orthogonal_complement(seed):
    rng = np.random.default_rng(seed)
    width = int(rng.integers(1, 150))
    vectors = rng.integers(0, 2, (int(rng.integers(0, width + 5)), width))
    vectors = (vectors * (rng.random(vectors.shape) < 0.3)).astype(np.uint8)
    span = Span(width)
    grew = span.insert_bits(vectors)
    assert span.rank == grew.sum() == _rank(vectors)
    annihilator = unpack(span.annihilator(), width)
    assert len(annihilator) == width - span.rank
    assert not ((vectors.astype(int) @ annihilator.T.astype(int)) % 2).any()
    assert _rank(annihilator) == len(annihilator)
    ends = [int(np.flatnonzero(row).max()) for row in annihilator]
    assert ends == sorted(set(ends))


def test_the_basis_has_one_vector_per_pivot():
    span = Span(3)
    grew = span.insert(pack(np.array([[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]])))
    assert grew.tolist() == [False, True, True, False]
    assert span.rank == 2
    assert unpack(span.basis(), 3).tolist() == [[1, 1, 0], [0, 1, 1]]
