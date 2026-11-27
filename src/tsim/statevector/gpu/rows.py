"""The rows of outputs that the GPU backends return."""

from __future__ import annotations

import numpy as np


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
