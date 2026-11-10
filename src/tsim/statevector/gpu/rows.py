"""The rows of outputs that the GPU backends return, and how their kernels pack them."""

from __future__ import annotations

import numpy as np


def pieces(columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return how a kernel packs the outputs ``columns`` of shot words: ``(pieces, start)``.

    Output word ``k`` holds ``columns[64 k : 64 k + 64]``; its pieces are
    ``pieces[start[k] : start[k + 1]]``, each a run of them that is consecutive
    in one shot word, as ``(word, bit | (length - 1) << 6 | position << 12)``.
    """
    found, start = [], [0]
    for k in range(0, len(columns), 64):
        word = columns[k : k + 64].tolist()
        i = 0
        while i < len(word):
            j = i
            while (
                j + 1 < len(word)
                and word[j + 1] == word[j] + 1
                and word[j + 1] >> 6 == word[i] >> 6
            ):
                j += 1
            c = word[i]
            found.append((c >> 6, (c & 63) | (j - i) << 6 | i << 12))
            i = j + 1
        start.append(len(found))
    return np.asarray(found or [(0, 0)], np.uint32), np.asarray(start, np.uint32)


def select(
    rows: np.ndarray, columns: np.ndarray, num_outputs: int, bit_packed: bool
) -> np.ndarray:
    """Return the outputs ``columns`` of rows of every output, as ``read_outputs`` does.

    The rows are bit-packed as ``np.packbits`` does, with zero bits past the
    outputs; leading outputs in order are a view of them, or of whole bytes.
    """
    count = len(columns)
    if bit_packed and np.array_equal(columns, np.arange(count)):
        packed = rows[:, : -(-count // 8)]
        if count == num_outputs or count % 8 == 0:
            return packed
        packed = packed.copy()
        packed[:, -1] &= np.uint8((1 << (count % 8)) - 1)
        return packed
    bits = np.unpackbits(rows, axis=1, count=num_outputs, bitorder="little")
    bits = bits.view(np.bool_)[:, columns]
    return np.packbits(bits, axis=1, bitorder="little") if bit_packed else bits
