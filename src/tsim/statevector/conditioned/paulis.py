"""Pauli rows on bit words, and Clifford gates on them.

A row of ``2 * W + 1`` uint64 words stores the Pauli ``i^e X^x Z^z`` on ``n <= 64 W``
qubits: words ``0 … W-1`` hold ``x``, words ``W … 2W-1`` hold ``z`` and the last
word holds ``e`` (mod 4). A Hermitian Pauli with letters has ``e = #Y + 2 [sign < 0]``.
In this form a product needs one popcount for its phase:

    (i^a X^x Z^z)(i^b X^u Z^v) = i^(a + b + 2 |z & u|) X^(x ^ u) Z^(z ^ v).

A frame (a Clifford ``F``) is a table of ``2n`` rows: row ``q`` is ``F X_q F^dag``
and row ``n + q`` is ``F Z_q F^dag``. Gates act on the virtual qubits, that is on
the columns of every row, which conjugates the rows: ``R -> G R G^dag``.
"""

from __future__ import annotations

import numpy as np

from tsim.utils.kernels import jit

_ONE = np.uint64(1)
_THREE = np.uint64(3)


@jit
def popcount(value: np.uint64) -> np.int64:
    """Return the number of set bits of ``value``."""
    v = value - ((value >> np.uint64(1)) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + (
        (v >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((v * np.uint64(0x0101010101010101)) >> np.uint64(56))


@jit
def get_bit(row, word_offset: int, q: int) -> np.int64:
    """Return bit ``q`` of the bit vector that starts at word ``word_offset`` of ``row``."""
    return np.int64((row[word_offset + (q >> 6)] >> np.uint64(q & 63)) & _ONE)


@jit
def set_identity(row, num_words: int) -> None:
    """Set ``row`` to the identity."""
    for w in range(2 * num_words + 1):
        row[w] = 0


@jit
def copy_row(dst, src, num_words: int) -> None:
    """Copy ``src`` into ``dst``."""
    for w in range(2 * num_words + 1):
        dst[w] = src[w]


@jit
def multiply_right(dst, src, num_words: int) -> None:
    """Set ``dst`` to ``dst * src``."""
    twists = 0
    for w in range(num_words):
        twists += popcount(dst[num_words + w] & src[w])
        dst[w] ^= src[w]
        dst[num_words + w] ^= src[num_words + w]
    dst[2 * num_words] = (
        dst[2 * num_words] + src[2 * num_words] + np.uint64(2 * twists)
    ) & _THREE


@jit
def anticommutes(a, b, num_words: int) -> bool:
    """Return whether the Paulis ``a`` and ``b`` anticommute."""
    count = 0
    for w in range(num_words):
        count += popcount((a[w] & b[num_words + w]) ^ (a[num_words + w] & b[w]))
    return (count & 1) == 1


@jit
def image(frame, x, z, e: int, out, num_words: int) -> None:
    """Write ``F (i^e X^x Z^z) F^dag`` to ``out``, for x and z as bit words."""
    n = frame.shape[0] // 2
    set_identity(out, num_words)
    out[2 * num_words] = np.uint64(e & 3)
    for w in range(num_words):
        bits = x[w]
        while bits:
            low = bits & (~bits + _ONE)
            q = 64 * w + popcount(low - _ONE)
            multiply_right(out, frame[q], num_words)
            bits ^= low
    for w in range(num_words):
        bits = z[w]
        while bits:
            low = bits & (~bits + _ONE)
            q = 64 * w + popcount(low - _ONE)
            multiply_right(out, frame[n + q], num_words)
            bits ^= low


# -- Clifford gates on the columns of rows ------------------------------------
# Each gate G maps every row R to G R G^dag. The rules follow from conjugating
# X and Z and reordering into the form i^e X^x Z^z.


@jit
def gate_h(rows, q: int, num_words: int) -> None:
    """Apply H on qubit ``q``: X <-> Z, and XZ -> -XZ."""
    w, b = q >> 6, np.uint64(q & 63)
    for r in range(rows.shape[0]):
        x = (rows[r, w] >> b) & _ONE
        z = (rows[r, num_words + w] >> b) & _ONE
        if x != z:
            rows[r, w] ^= _ONE << b
            rows[r, num_words + w] ^= _ONE << b
        elif x:
            rows[r, 2 * num_words] = (rows[r, 2 * num_words] + np.uint64(2)) & _THREE


@jit
def gate_s(rows, q: int, num_words: int, dagger: bool) -> None:
    """Apply S (or S^dag) on qubit ``q``: X -> +-iXZ, Z -> Z."""
    w, b = q >> 6, np.uint64(q & 63)
    phase = np.uint64(3) if dagger else np.uint64(1)
    for r in range(rows.shape[0]):
        if (rows[r, w] >> b) & _ONE:
            rows[r, num_words + w] ^= _ONE << b
            rows[r, 2 * num_words] = (rows[r, 2 * num_words] + phase) & _THREE


@jit
def gate_pauli(rows, q: int, num_words: int, letter: int) -> None:
    """Apply the Pauli X (letter 1), Y (3) or Z (2) on qubit ``q``: signs of anticommuting rows flip."""
    w, b = q >> 6, np.uint64(q & 63)
    for r in range(rows.shape[0]):
        x = int((rows[r, w] >> b) & _ONE)
        z = int((rows[r, num_words + w] >> b) & _ONE)
        anti = ((letter & 1) * z) ^ ((letter >> 1) * x)
        if anti:
            rows[r, 2 * num_words] = (rows[r, 2 * num_words] + np.uint64(2)) & _THREE


@jit
def gate_cx(rows, c: int, t: int, num_words: int) -> None:
    """Apply CX: X_c -> X_c X_t, Z_t -> Z_c Z_t (no phase in this form)."""
    wc, bc = c >> 6, np.uint64(c & 63)
    wt, bt = t >> 6, np.uint64(t & 63)
    for r in range(rows.shape[0]):
        if (rows[r, wc] >> bc) & _ONE:
            rows[r, wt] ^= _ONE << bt
        if (rows[r, num_words + wt] >> bt) & _ONE:
            rows[r, num_words + wc] ^= _ONE << bc


@jit
def gate_cz(rows, a: int, b: int, num_words: int) -> None:
    """Apply CZ: X_a -> X_a Z_b, X_b -> X_b Z_a, and -1 when both X parts are set."""
    wa, ba = a >> 6, np.uint64(a & 63)
    wb, bb = b >> 6, np.uint64(b & 63)
    for r in range(rows.shape[0]):
        xa = (rows[r, wa] >> ba) & _ONE
        xb = (rows[r, wb] >> bb) & _ONE
        if xa:
            rows[r, num_words + wb] ^= _ONE << bb
        if xb:
            rows[r, num_words + wa] ^= _ONE << ba
        if xa & xb:
            rows[r, 2 * num_words] = (rows[r, 2 * num_words] + np.uint64(2)) & _THREE
