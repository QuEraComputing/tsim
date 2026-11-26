"""Lists the pieces of a circuit's faults: minimal sets whose direct detectors cancel.

Every set of faults whose direct detectors cancel is a disjoint union of
pieces, the nonempty such sets with no smaller nonempty such subset. ``Pieces``
lists those of ``k`` faults by canonical branching, the weight-bounded search
of Cheng, Guo and Pan's distance certifier (arXiv:2611.08247), after Dumer:

- A piece is reached from its lowest fault, the anchor. While some detectors
  stay flipped, the search takes the flipped detector that the fewest free
  faults flip and adds each of those faults in turn; a fault it has added
  stops being free for the ones after it, so each piece is reached once.
- ``k - size`` more faults flip at most ``most * (k - size)`` detectors, which
  prunes the search.
- The last fault, or the last two when there are few enough pairs, come from
  tables of the faults by their detectors, through a GF(2)-linear hash.

The search stops only at sets whose detectors cancel, so the faults it adds
after the anchor may form two or more pieces; such a set is listed too, and its
detectors cancel as well.

Each fault also has a value, a row of words, and a set's value is the XOR of
its faults'. The search tracks it along the way, so that it can return the
values of the pieces, or only the pieces whose values are given, without
writing every piece.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
from math import comb

import numba
import numpy as np

from tsim.noise.fault_values import find, index, pack
from tsim.utils.kernels import jit, kernel

MAX_PAIRS = 1 << 24
"""Most pairs of faults in the table of pairs; above, only single faults are."""

BATCH_BYTES = 1 << 30
"""Most bytes of values that ``Pieces.values`` yields at once, unless one anchor
has more."""

COUNT, VALUES, MATCHING = 0, 1, 2
"""What the search does with each piece: count it, write its value, or write its
faults if its value is a target."""

_M3 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_M5 = np.uint64(0x5555555555555555)
_ONES = np.uint64(0x0101010101010101)


@jit
def _popcount(x):
    """Return the number of bits set in the uint64 ``x``."""
    x = x - ((x >> np.uint64(1)) & _M5)
    x = (x & _M3) + ((x >> np.uint64(2)) & _M3)
    x = (x + (x >> np.uint64(4))) & _M4
    return int((x * _ONES) >> np.uint64(56))


@jit
def _pairs(detectors, faults):
    """Return the pairs of ``faults`` whose detectors differ, as two columns."""
    m = len(faults)
    first = np.empty(m * (m - 1) // 2, dtype=np.int64)
    second = np.empty_like(first)
    n = 0
    for i in range(m):
        for j in range(i + 1, m):
            same = True
            for w in range(detectors.shape[1]):
                if detectors[faults[i], w] != detectors[faults[j], w]:
                    same = False
                    break
            if not same:
                first[n] = faults[i]
                second[n] = faults[j]
                n += 1
    return first[:n], second[:n]


@jit
def _is_free(f, a, forbidden):
    """Return whether the search of anchor ``a`` may still add fault ``f``."""
    return f > a and forbidden[f] == 0


@jit
def _emit(sink, found, path, size, extra, value):
    """Handle the piece ``path[:size]`` and ``extra``, but -1; return the new ``found``.

    ``value`` is the value of ``path[:size]``. ``sink`` is ``(mode, out,
    words, targets, scratch)``: what to do, the rows to write from ``found``
    on, the faults' values, ``(index, rows)`` of the target values, and room
    for one value.
    """
    mode, out, words, targets, scratch = sink
    if mode == COUNT:
        return found + 1
    for k in range(len(scratch)):
        scratch[k] = value[k]
        for g in extra:
            if g >= 0:
                scratch[k] ^= words[g, k]
    if mode == VALUES:
        out[found, : len(scratch)] = scratch
        return found + 1
    if find(targets[0], targets[1], scratch) < 0:
        return found
    for t in range(size):
        out[found, t] = path[t]
    for g in extra:
        if g >= 0:
            out[found, size] = g
            size += 1
    return found + 1


@jit
def _complete(a, d, left, state, faults, forbidden, tables, sink, found):
    """Handle each completion of ``path[:d + 1]`` by ``left`` faults; return ``found``.

    A completion is one free fault (``left == 1``) or pair of them whose
    detectors are ``sigma[d]``.
    """
    path, sigma, hashes_at, values_at, _, _, extra = state
    detectors = faults[0]
    singles, pairs = tables
    keys, key_index, start = singles[0], singles[1], singles[2]
    if left == 2:
        keys, key_index, start = pairs[0], pairs[1], pairs[2]
    i = find(key_index, keys, hashes_at[d : d + 1])
    if i < 0:
        return found
    extra[:] = -1
    for q in range(start[i], start[i + 1]):
        if left == 1:
            extra[0] = singles[3][q]
        else:
            extra[0], extra[1] = pairs[3][q], pairs[4][q]
        usable = True
        for g in extra:
            if g >= 0 and not _is_free(g, a, forbidden):
                usable = False
        for w in range(detectors.shape[1]):
            if not usable:
                break
            x = sigma[d, w]
            for g in extra:
                if g >= 0:
                    x ^= detectors[g, w]
            usable = x == 0
        if usable:
            found = _emit(sink, found, path, d + 1, extra, values_at[d])
    return found


@jit
def _anchor(a, k, depth, faults, forbidden, tables, sink, state):
    """Handle the pieces of ``k`` faults whose lowest is ``a``; return their number.

    The search adds faults one by one up to ``depth`` faults, then completes
    the set from the tables. It leaves ``forbidden`` as it found it, all 0.
    """
    detectors, hashes, row_start, row_faults, most, words = faults
    path, sigma, hashes_at, values_at, row_at, next_at, extra = state
    num_words = detectors.shape[1]
    found = 0
    path[0] = a
    sigma[0] = detectors[a]
    hashes_at[0] = hashes[a]
    values_at[0] = words[a]
    row_at[0] = -1
    d = 0
    while d >= 0:
        if row_at[d] < 0:
            # NOTE: the first visit of the set ``path[:d + 1]``.
            size = d + 1
            left = k - size
            flipped = 0
            for w in range(num_words):
                flipped += _popcount(sigma[d, w])
            if flipped == 0 or left == 0 or flipped > most * left:
                # NOTE: a set whose detectors cancel ends the branch: adding
                # faults to it cannot give a piece.
                if flipped == 0 and left == 0:
                    extra[:] = -1
                    found = _emit(sink, found, path, size, extra, values_at[d])
                d -= 1
                continue
            if size == depth:
                found = _complete(
                    a, d, left, state, faults, forbidden, tables, sink, found
                )
                d -= 1
                continue
            # NOTE: branch on the flipped detector with the fewest free faults.
            best, choice = 1 << 62, -1
            for w in range(num_words):
                x = sigma[d, w]
                while x:
                    low = x & (~x + np.uint64(1))
                    r = w * 64 + _popcount(low - np.uint64(1))
                    x ^= low
                    free = 0
                    for q in range(row_start[r], row_start[r + 1]):
                        if _is_free(row_faults[q], a, forbidden):
                            free += 1
                    if free < best:
                        best, choice = free, r
            if best == 0:
                d -= 1
                continue
            row_at[d] = choice
            next_at[d] = row_start[choice]
        r = row_at[d]
        f = -1
        while next_at[d] < row_start[r + 1]:
            g = row_faults[next_at[d]]
            next_at[d] += 1
            if _is_free(g, a, forbidden):
                f = g
                break
        if f < 0:
            # NOTE: every fault of the row is tried; free them for the parent.
            for q in range(row_start[r], row_start[r + 1]):
                if forbidden[row_faults[q]] == d + 1:
                    forbidden[row_faults[q]] = 0
            row_at[d] = -1
            d -= 1
            continue
        forbidden[f] = d + 1
        # NOTE: loops, not array expressions, which would allocate.
        for w in range(num_words):
            sigma[d + 1, w] = sigma[d, w] ^ detectors[f, w]
        for w in range(words.shape[1]):
            values_at[d + 1, w] = values_at[d, w] ^ words[f, w]
        hashes_at[d + 1] = hashes_at[d] ^ hashes[f]
        path[d + 1] = f
        row_at[d + 1] = -1
        d += 1
    return found


@jit
def _lane(
    lane, lanes, k, depth, faults, tables, targets, mode, anchors, counts, offsets, out
):
    """Run the search of the anchors ``anchors[lane::lanes]``."""
    detectors, words = faults[0], faults[5]
    forbidden = np.zeros(len(detectors), dtype=np.int32)
    state = (
        np.zeros(k, dtype=np.int64),
        np.zeros((k, detectors.shape[1]), dtype=np.uint64),
        np.zeros(k, dtype=np.uint64),
        np.zeros((k, words.shape[1]), dtype=np.uint64),
        np.zeros(k, dtype=np.int64),
        np.zeros(k, dtype=np.int64),
        np.full(2, -1, dtype=np.int64),
    )
    scratch = np.empty(words.shape[1], dtype=np.uint64)
    for t in range(lane, len(anchors), lanes):
        a = anchors[t]
        sink = (
            mode,
            out[offsets[a] :] if mode != COUNT else out,
            words,
            targets,
            scratch,
        )
        counts[a] = _anchor(a, k, depth, faults, forbidden, tables, sink, state)


def _search(
    k,
    depth,
    faults,
    singles,
    pairs,
    target_index,
    targets,
    mode,
    anchors,
    lanes,
    counts,
    offsets,
    out,
):
    """Handle the pieces of ``k`` faults whose lowest faults are ``anchors``.

    ``counts[a]`` becomes the number of pieces of anchor ``a`` that the search
    counted, or wrote from row ``offsets[a]`` of ``out`` on. Lane ``lane``
    takes every ``lanes``-th anchor from ``anchors[lane]`` on, which mixes
    cheap and costly ones.
    """
    # NOTE: numba's parallel loops take flat tuples of arrays only.
    for lane in numba.prange(lanes):
        _lane(
            lane,
            lanes,
            k,
            depth,
            faults,
            (singles, pairs),
            (target_index, targets),
            mode,
            anchors,
            counts,
            offsets,
            out,
        )


search = kernel(_search)
"""``search(work, k, depth, faults, singles, pairs, target_index, targets, mode,
anchors, lanes, counts, offsets, out)``, in parallel when ``work`` is large."""


class Pieces:
    """The faults' direct detectors and values, indexed to list their pieces."""

    def __init__(self, detectors: np.ndarray, values: np.ndarray) -> None:
        """Index ``detectors[f, d]``, whether fault ``f`` flips direct detector ``d``.

        ``values[f]`` is the value of fault ``f``, a row of uint64 words.
        """
        detectors = np.asarray(detectors, dtype=bool)
        n, m = detectors.shape
        self._n = n
        rows, faults = np.nonzero(detectors.T)
        self._row_start = np.searchsorted(rows, np.arange(m + 1)).astype(np.int64)
        self._row_faults = faults.astype(np.int64)
        # NOTE: a GF(2)-linear hash of the detectors, so that a set's hash is
        # the XOR of its faults' hashes.
        mixer = np.random.default_rng(0).integers(0, 2**63, size=m, dtype=np.uint64)
        hashes = np.zeros(n, dtype=np.uint64)
        np.bitwise_xor.at(hashes, faults, mixer[rows])
        self._faults = (
            pack(detectors),
            hashes,
            self._row_start,
            self._row_faults,
            int(detectors.sum(axis=1).max(initial=0)),
            np.ascontiguousarray(values, dtype=np.uint64),
        )
        # NOTE: a fault that flips no detector is a piece alone, so no other
        # piece holds it.
        self._flipping = np.flatnonzero(detectors.any(axis=1))
        self._singles = _table(hashes[self._flipping], (self._flipping,))
        self._counts: dict[int, np.ndarray] = {}

    def counts(self, k: int, limit: int) -> np.ndarray | None:
        """Return the number of pieces of ``k`` faults of each lowest fault.

        Returns None, before the search ends, once they are more than ``limit``.
        """
        if k not in self._counts:
            counts = np.zeros(self._n, dtype=np.int64)
            empty = np.zeros((0, 0), dtype=np.uint64)
            # NOTE: anchors go in chunks, to stop a search that grows too large.
            step = 256 * numba.get_num_threads()
            for lo in range(0, self._n, step):
                anchors = np.arange(lo, min(lo + step, self._n))
                self._search(k, COUNT, anchors, counts, counts, empty)
                if counts.sum() > limit:
                    return None
            self._counts[k] = counts
        return self._counts[k]

    def values(self, k: int) -> Iterator[np.ndarray]:
        """Yield the values of the pieces of ``k`` faults, in batches of ``BATCH_BYTES``.

        ``counts`` must have returned their number first.
        """
        counts = self._counts[k]
        for anchors, offsets, total in self._batches(counts):
            out = np.empty((total, self._faults[5].shape[1]), dtype=np.uint64)
            written = np.zeros(self._n, dtype=np.int64)
            self._search(k, VALUES, anchors, written, offsets, out)
            assert np.array_equal(written[anchors], counts[anchors])
            yield out

    def matching(self, k: int, targets: np.ndarray) -> np.ndarray:
        """Return the pieces of ``k`` faults whose values are among ``targets``.

        Each piece is a sorted row of fault indices. ``targets`` are distinct,
        and ``counts`` must have returned the number of pieces first.
        """
        counts = self._counts[k]
        parts = [np.zeros((0, k), dtype=np.int32)]
        target_table = (index(targets), targets)
        for anchors, offsets, total in self._batches(counts):
            # NOTE: an anchor's matching pieces go to the start of its rows,
            # and only they are written.
            out = np.empty((total, k), dtype=np.uint64)
            written = np.zeros(self._n, dtype=np.int64)
            self._search(k, MATCHING, anchors, written, offsets, out, target_table)
            mine = written[anchors]
            assert (mine <= counts[anchors]).all()
            start = np.repeat(offsets[anchors] - (np.cumsum(mine) - mine), mine)
            rows = out[start + np.arange(len(start))].astype(np.int32)
            parts.append(np.sort(rows, axis=1))
        return np.concatenate(parts)

    def _batches(
        self, counts: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        """Yield anchors with about ``BATCH_BYTES`` of values, their offsets and total.

        A batch takes every ``m``-th anchor, so that it mixes cheap and costly
        ones as a range of anchors would not, and deals them out to the lanes
        by their counts, largest first.
        """
        lanes = numba.get_num_threads()
        size = 8 * self._faults[5].shape[1]
        m = max(1, min(self._n, -(-int(counts.sum()) * size // BATCH_BYTES)))
        for first in range(m):
            anchors = np.arange(first, self._n, m)
            # NOTE: lane ``i`` takes ``anchors[i::lanes]``; snake order evens
            # out their work.
            anchors = anchors[np.argsort(-counts[anchors], kind="stable")]
            for start in range(lanes, len(anchors), 2 * lanes):
                anchors[start : start + lanes] = anchors[start : start + lanes][::-1]
            offsets = np.zeros(self._n, dtype=np.int64)
            offsets[anchors] = np.cumsum(counts[anchors]) - counts[anchors]
            yield anchors, offsets, int(counts[anchors].sum())

    def _search(
        self,
        k: int,
        mode: int,
        anchors: np.ndarray,
        counts: np.ndarray,
        offsets: np.ndarray,
        out: np.ndarray,
        targets: tuple | None = None,
    ) -> None:
        """Run ``search`` on ``anchors``."""
        pairs = self._pairs if k >= 3 else None
        # NOTE: the tables complete the last fault, or the last two if pairs fit.
        depth = max(1, k - 1 if pairs is None else k - 2)
        if targets is None:
            none = np.zeros((0, self._faults[5].shape[1]), dtype=np.uint64)
            targets = (index(none), none)
        tables = (self._singles, _NO_PAIRS if pairs is None else pairs)
        lanes = max(1, min(numba.get_num_threads(), len(anchors)))
        work = len(anchors) * len(self._row_faults)
        args = (k, depth, self._faults, *tables, *targets, mode, anchors, lanes)
        search(work, *args, counts, offsets, out)

    @cached_property
    def _pairs(self) -> tuple | None:
        """The table of pairs of faults, or None if there are too many."""
        if comb(len(self._flipping), 2) > MAX_PAIRS:
            return None
        detectors, hashes = self._faults[0], self._faults[1]
        first, second = _pairs(detectors, self._flipping)
        return _table(hashes[first] ^ hashes[second], (first, second))


def _table(keys: np.ndarray, columns: tuple[np.ndarray, ...]) -> tuple:
    """Return the distinct ``keys`` as rows, their ``index``, and ``columns`` by key.

    The entries of ``columns`` of key ``i`` are ``start[i]`` to
    ``start[i + 1] - 1``.
    """
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    change = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    start = np.concatenate([[0], change, [len(keys)]]).astype(np.int64)
    if not len(keys):
        start = start[:1]
    distinct = np.ascontiguousarray(ordered[start[:-1]].reshape(-1, 1))
    grouped = tuple(np.ascontiguousarray(c[order], dtype=np.int64) for c in columns)
    return (distinct, index(distinct), start, *grouped)


_NO_PAIRS = _table(np.zeros(0, dtype=np.uint64), (np.zeros(0), np.zeros(0)))
