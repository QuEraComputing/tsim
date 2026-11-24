"""Tests for ``tsim.noise.fault_pieces``: the pieces of a set of faults."""

import itertools

import numpy as np
import pytest

from tsim.noise import fault_pieces
from tsim.noise.fault_pieces import Pieces
from tsim.noise.fault_values import pack


def _cancels(detectors: np.ndarray, faults: tuple) -> bool:
    return not np.bitwise_xor.reduce(detectors[list(faults)], axis=0).any()


def _sets(detectors: np.ndarray, k: int) -> tuple[set, set]:
    """Return the sets of ``k`` faults whose detectors cancel, and the pieces."""
    every, pieces = set(), set()
    for faults in itertools.combinations(range(len(detectors)), k):
        if _cancels(detectors, faults):
            every.add(faults)
            smaller = (
                sub for r in range(1, k) for sub in itertools.combinations(faults, r)
            )
            if not any(_cancels(detectors, sub) for sub in smaller):
                pieces.add(faults)
    return every, pieces


def _faults(values: np.ndarray, n: int) -> list[tuple]:
    """Return the faults that one-hot values name."""
    bits = np.unpackbits(values.view(np.uint8), axis=1, bitorder="little")[:, :n]
    return [tuple(np.flatnonzero(row).tolist()) for row in bits]


def _detectors(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(3, 12)), int(rng.integers(0, 6))
    detectors = rng.random((n, m)) < 0.35
    # NOTE: two faults with the same detectors.
    detectors[rng.integers(n)] = detectors[rng.integers(n)]
    return detectors


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize(("batch", "max_pairs"), [(1 << 30, 1 << 24), (3, 0)])
def test_every_piece_is_listed_once(seed, batch, max_pairs, monkeypatch):
    monkeypatch.setattr(fault_pieces, "BATCH_BYTES", batch)
    monkeypatch.setattr(fault_pieces, "MAX_PAIRS", max_pairs)
    detectors = _detectors(seed)
    n = len(detectors)
    # NOTE: one-hot values, so that a piece's value names its faults.
    pieces = Pieces(detectors, pack(np.eye(n, dtype=np.uint8)))
    for k in range(1, 6):
        counts = pieces.counts(k, limit=1 << 40)
        assert counts is not None
        listed = [f for values in pieces.values(k) for f in _faults(values, n)]
        every, expected = _sets(detectors, k)
        assert len(listed) == len(set(listed)) == counts.sum()
        assert expected <= set(listed) <= every


@pytest.mark.parametrize("seed", range(10))
def test_matching_lists_the_pieces_with_the_target_values(seed):
    detectors = _detectors(seed)
    n = len(detectors)
    values = pack(np.eye(n, dtype=np.uint8))
    pieces = Pieces(detectors, values)
    for k in range(1, 5):
        pieces.counts(k, limit=1 << 40)
        every = np.concatenate([np.zeros((0, 1), np.uint64), *pieces.values(k)])
        targets = every[::2]
        got = pieces.matching(k, targets)
        assert sorted(map(tuple, got.tolist())) == sorted(_faults(targets, n))


def test_counting_stops_past_the_limit():
    detectors = np.zeros((4, 1), dtype=bool)
    pieces = Pieces(detectors, pack(np.eye(4, dtype=np.uint8)))
    assert pieces.counts(1, limit=3) is None
    assert pieces.counts(1, limit=4) is not None
