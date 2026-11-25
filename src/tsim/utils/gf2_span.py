"""A growing span of GF(2) vectors, packed into 64-bit words.

``Span`` keeps one basis vector per pivot, the lowest set bit of that vector,
so inserting a vector reduces it by at most one basis vector per pivot, as
``linalg.find_basis`` does with Python integers. Its annihilator, the vectors
orthogonal to the whole span, then has one basis vector per column that is no
pivot: that column plus the pivots before it.
"""

from __future__ import annotations

import numba
import numpy as np


def num_words(width: int) -> int:
    """Return the 64-bit words of a packed vector of ``width`` bits."""
    return max(1, -(-width // 64))


def pack(bits: np.ndarray) -> np.ndarray:
    """Return rows of 0/1 entries as rows of little-endian 64-bit words."""
    bits = np.asarray(bits, dtype=np.uint8)
    rows, width = bits.shape
    padded = np.zeros((rows, num_words(width) * 64), dtype=np.uint8)
    padded[:, :width] = bits
    packed = np.packbits(padded, axis=1, bitorder="little")
    return packed.view("<u8").reshape(rows, num_words(width))


def pack_bytes(rows: np.ndarray, width: int) -> np.ndarray:
    """Return little-endian bit-packed bytes, ``width`` bits a row, as 64-bit words."""
    padded = np.zeros((len(rows), num_words(width) * 8), dtype=np.uint8)
    padded[:, : rows.shape[1]] = rows
    return padded.view("<u8").reshape(len(rows), num_words(width))


def unpack(words: np.ndarray, width: int) -> np.ndarray:
    """Return rows of 64-bit words as rows of ``width`` 0/1 entries."""
    as_bytes = np.ascontiguousarray(words).view(np.uint8)
    return np.unpackbits(as_bytes, axis=1, bitorder="little")[:, :width]


@numba.njit(cache=True)
def _lowest(row: np.ndarray) -> int:
    for w in range(len(row)):
        word = row[w]
        if word:
            bit = 0
            while not (word >> np.uint64(bit)) & np.uint64(1):
                bit += 1
            return 64 * w + bit
    return -1


@numba.njit(cache=True)
def _has_bit(row: np.ndarray, column: int) -> bool:
    return bool((row[column // 64] >> np.uint64(column % 64)) & np.uint64(1))


@numba.njit(cache=True)
def _insert(basis: np.ndarray, has: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Insert ``rows`` and return, for each, whether it grew the span."""
    grew = np.zeros(len(rows), dtype=np.bool_)
    v = np.empty(basis.shape[1], dtype=np.uint64)
    for i in range(len(rows)):
        v[:] = rows[i]
        while True:
            p = _lowest(v)
            if p < 0:
                break
            if has[p]:
                v ^= basis[p]
            else:
                basis[p] = v
                has[p] = True
                grew[i] = True
                break
    return grew


@numba.njit(cache=True)
def _annihilator(basis: np.ndarray, has: np.ndarray) -> np.ndarray:
    """Return, per column j that is no pivot, e_j plus the pivots before it, packed."""
    width = len(has)
    reduced = basis.copy()
    # NOTE: clearing each pivot column from the other basis vectors, from the
    # last pivot down, leaves each with its pivot and columns that are none.
    for p in range(width - 1, -1, -1):
        if has[p]:
            for q in range(p):
                if has[q] and _has_bit(reduced[q], p):
                    reduced[q] ^= reduced[p]
    slot = np.full(width, -1, dtype=np.int64)
    count = 0
    for j in range(width):
        if not has[j]:
            slot[j] = count
            count += 1
    result = np.zeros((count, basis.shape[1]), dtype=np.uint64)
    for j in range(width):
        if slot[j] >= 0:
            result[slot[j], j // 64] |= np.uint64(1) << np.uint64(j % 64)
    for p in range(width):
        if has[p]:
            for j in range(p + 1, width):
                if slot[j] >= 0 and _has_bit(reduced[p], j):
                    result[slot[j], p // 64] |= np.uint64(1) << np.uint64(p % 64)
    return result


class Span:
    """The span of the vectors inserted so far, in ``GF(2)^width``."""

    def __init__(self, width: int) -> None:
        """Start with the zero span."""
        self.width = width
        self._basis = np.zeros((width, num_words(width)), dtype=np.uint64)
        self._has = np.zeros(width, dtype=np.bool_)

    @property
    def rank(self) -> int:
        """Return the dimension of the span."""
        return int(self._has.sum())

    def basis(self) -> np.ndarray:
        """Return a basis of the span, packed."""
        return self._basis[self._has]

    def insert(self, words: np.ndarray) -> np.ndarray:
        """Insert packed vectors and return, for each, whether it grew the span."""
        return _insert(self._basis, self._has, self._rows(words))

    def insert_bits(self, bits: np.ndarray) -> np.ndarray:
        """Insert vectors of 0/1 entries; see ``insert``."""
        bits = np.asarray(bits, dtype=np.uint8)
        return self.insert(pack(bits[None] if bits.ndim == 1 else bits))

    def annihilator(self) -> np.ndarray:
        """Return a basis of the vectors orthogonal to the span, packed.

        Row ``i`` is a column ``j`` that is no pivot plus pivots before ``j``,
        so the rows end at distinct columns, in increasing order.
        """
        return _annihilator(self._basis, self._has)

    def _rows(self, words: np.ndarray) -> np.ndarray:
        words = np.ascontiguousarray(words, dtype=np.uint64)
        return words.reshape(-1, num_words(self.width))
