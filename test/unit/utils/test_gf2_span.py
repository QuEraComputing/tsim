"""Tests for ``tsim.utils.gf2_span``."""

import numpy as np
import pytest

from tsim.utils.gf2_span import Span, pack, pack_bytes, unpack


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


def test_contains():
    span = Span(3)
    vectors = pack(np.array([[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]]))
    assert span.contains(vectors).tolist() == [True, False, False, False]
    span.insert(vectors[1])
    assert span.contains(vectors).tolist() == [True, True, False, False]
    span.insert(vectors[2])
    assert span.contains(vectors).tolist() == [True, True, True, True]
    assert span.rank == 2
    assert unpack(span.basis(), 3).tolist() == [[1, 1, 0], [0, 1, 1]]
