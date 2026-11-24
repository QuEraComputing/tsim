"""Numba kernels on the values of sets of faults.

A set of faults acts through the rotation signs it flips and the flips of the
outputs of components and of the direct observables, the XOR of its faults'.
Its value is that row of uint64 words: the first ``ws`` words are the signs,
the others the output flips. A value is a logical error when one of the
deviations that its signs can cause, XOR its output flips, flips no detector
and flips an observable.

The deviation table is a tuple ``(index, signs, start, deviations,
probability, detector_mask, observable_mask)``: the deviations of the walked
signs ``signs[t]`` are the rows ``start[t]`` to ``start[t + 1] - 1`` of
``deviations``, as words over the output flips, and ``index`` finds a sign.

Kernels that write a variable number of rows per task run twice: once with no
room for rows, to count them, then with room for them all.
"""

from __future__ import annotations

import numba
import numpy as np

from tsim.utils.kernels import jit, kernel

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)

_UNDETECTED = 1
"""What ``_outcome`` returns for a deviation that flips no detector or observable."""

_FLIPPED = 2
"""What ``_outcome`` returns for a deviation that flips an observable undetected."""


def pack(bits: np.ndarray) -> np.ndarray:
    """Return rows of bits as rows of uint64 words, at least one word per row."""
    width = max(1, -(-bits.shape[1] // 64))
    padded = np.zeros((len(bits), 64 * width), dtype=np.uint8)
    padded[:, : bits.shape[1]] = bits
    return np.packbits(padded, axis=1, bitorder="little").view(np.uint64)


@jit
def _mix(x):
    """Return the splitmix64 finalizer of ``x``."""
    x = (x ^ (x >> np.uint64(30))) * _M1
    x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


@jit
def _capacity(count):
    """Return the smallest power of 2 above twice ``count``, for a hash table."""
    size = 2
    while size <= 2 * count:
        size *= 2
    return size


@jit
def _hash(row):
    """Return a hash of a row of integers."""
    h = np.uint64(0x9E3779B97F4A7C15)
    for k in range(len(row)):
        h = _mix(h ^ np.uint64(row[k]))
    return h


@jit
def _equal(a, b):
    """Return whether two rows of the same length are equal."""
    same = True
    for k in range(len(a)):
        if a[k] != b[k]:
            same = False
            break
    return same


@jit
def _hash_table(rows, count):
    """Return an open-addressing table of ``rows[:count]``, with room for ``rows``."""
    table = np.full(_capacity(len(rows)), -1, dtype=np.int64)
    mask = np.uint64(len(table) - 1)
    for i in range(count):
        s = _hash(rows[i]) & mask
        while table[s] >= 0:
            s = (s + np.uint64(1)) & mask
        table[s] = i
    return table


@jit
def index(rows):
    """Return an open-addressing table of ``rows``, which must be distinct."""
    return _hash_table(rows, len(rows))


@jit
def find(table, rows, key):
    """Return the index of the row ``key`` in a table of ``index``, or -1."""
    mask = np.uint64(len(table) - 1)
    s = _hash(key) & mask
    while table[s] >= 0:
        if _equal(rows[table[s]], key):
            return table[s]
        s = (s + np.uint64(1)) & mask
    return -1


@jit
def distinct(rows, mask, inverse):
    """Return the first index of each distinct row of ``rows & mask``, in order.

    Also returns a table of ``index`` of those rows, ``rows[first] & mask``.
    Unless ``inverse`` is empty, ``inverse[i]`` becomes the number of the
    distinct row of row ``i``.
    """
    table = np.full(_capacity(len(rows)), -1, dtype=np.int64)
    slots = np.uint64(len(table) - 1)
    first = np.empty(len(rows), dtype=np.int64)
    key = np.empty(rows.shape[1], dtype=rows.dtype)
    other = np.empty_like(key)
    count = 0
    for i in range(len(rows)):
        for k in range(len(key)):
            key[k] = rows[i, k] & mask[k]
        s = _hash(key) & slots
        while True:
            t = table[s]
            if t < 0:
                table[s] = count
                first[count] = i
                t = count
                count += 1
                break
            for k in range(len(key)):
                other[k] = rows[first[t], k] & mask[k]
            if _equal(other, key):
                break
            s = (s + np.uint64(1)) & slots
        if len(inverse):
            inverse[i] = t
    return first[:count].copy(), table


@jit
def add_sums(rows, count, table, x, y, max_bytes):
    """Add the XORs of a row of ``x`` and a row of ``y`` to the distinct ``rows[:count]``.

    ``table`` indexes ``rows[:count]`` with room for ``rows``, or is empty to
    build it; both double as needed. Returns ``rows``, the new count and
    ``table``, or -1 for the count once they would take more than
    ``max_bytes``.
    """
    width = rows.shape[1]
    if len(table) == 0:
        if 8 * (len(rows) * width + _capacity(len(rows))) > max_bytes:
            return rows, -1, table
        table = _hash_table(rows, count)
    key = np.empty(width, dtype=np.uint64)
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(width):
                key[k] = x[i, k] ^ y[j, k]
            slots = np.uint64(len(table) - 1)
            s = _hash(key) & slots
            while table[s] >= 0 and not _equal(rows[table[s]], key):
                s = (s + np.uint64(1)) & slots
            if table[s] >= 0:
                continue
            if count == len(rows):
                size = 2 * len(rows)
                if 8 * (size * width + _capacity(size)) > max_bytes:
                    return rows, -1, table
                grown = np.empty((size, width), dtype=np.uint64)
                grown[:count] = rows[:count]
                rows = grown
                table = _hash_table(rows, count)
                slots = np.uint64(len(table) - 1)
                s = _hash(key) & slots
                while table[s] >= 0:
                    s = (s + np.uint64(1)) & slots
            rows[count] = key
            table[s] = count
            count += 1
    return rows, count, table


@jit
def group(inverse, count):
    """Return where each of ``count`` groups starts in ``order``, and ``order``.

    ``order`` lists the rows of group 0, then of group 1 and so on, each in
    increasing order; ``inverse[i]`` is the group of row ``i``.
    """
    start = np.zeros(count + 1, dtype=np.int64)
    for g in inverse:
        start[g + 1] += 1
    for g in range(count):
        start[g + 1] += start[g]
    position = start[:-1].copy()
    order = np.empty(len(inverse), dtype=np.int64)
    for i in range(len(inverse)):
        order[position[inverse[i]]] = i
        position[inverse[i]] += 1
    return start, order


def _find_all(table, rows, keys, out):
    """Write the index of each of ``keys`` in a table of ``index``, or -1, to ``out``."""
    for i in numba.prange(len(keys)):
        out[i] = find(table, rows, keys[i])


find_all = kernel(_find_all)
"""``find_all(work, table, rows, keys, out)``."""


def _xor_sets(sets, words, out):
    """Write the XOR of the rows ``words[sets[i]]`` to ``out[i]``."""
    for i in numba.prange(len(sets)):
        for k in range(words.shape[1]):
            x = np.uint64(0)
            for j in range(sets.shape[1]):
                x ^= words[sets[i, j], k]
            out[i, k] = x


xor_sets = kernel(_xor_sets)
"""``xor_sets(work, sets, words, out)``."""


@jit
def _outcome(flips, deviation, detector_mask, observable_mask):
    """Return what a deviation does to a set with output flips ``flips``.

    0 if it flips a detector, ``_UNDETECTED`` if it flips none and no
    observable, ``_FLIPPED`` if it flips no detector and an observable.
    """
    flipped = False
    for k in range(len(flips)):
        x = flips[k] ^ deviation[k]
        if x & detector_mask[k]:
            return 0
        if x & observable_mask[k]:
            flipped = True
    return _FLIPPED if flipped else _UNDETECTED


@jit
def _accepts(value, ws, table):
    """Return 1 if a value is a logical error, 0 if not, -1 if its signs were not walked."""
    sign_index, signs, start, deviations, _, detector_mask, observable_mask = table
    t = find(sign_index, signs, value[:ws])
    if t < 0:
        return -1
    for r in range(start[t], start[t + 1]):
        outcome = _outcome(value[ws:], deviations[r], detector_mask, observable_mask)
        if outcome == _FLIPPED:
            return 1
    return 0


def _accepted(values, ws, table, out):
    """Write whether each value is a logical error to ``out``, or -1 if not walked."""
    for i in numba.prange(len(values)):
        out[i] = _accepts(values[i], ws, table)


accepted = kernel(_accepted)
"""``accepted(work, values, ws, table, out)``."""


@jit
def _probability_of(value, ws, table):
    """Return a value's chance to flip no detector, and also an observable.

    Both are -1 if its signs were not walked.
    """
    sign_index, signs, start, deviations, probability, detector_mask, obs_mask = table
    t = find(sign_index, signs, value[:ws])
    if t < 0:
        return -1.0, -1.0
    undetected = flipped = 0.0
    for r in range(start[t], start[t + 1]):
        outcome = _outcome(value[ws:], deviations[r], detector_mask, obs_mask)
        if outcome:
            undetected += probability[r]
        if outcome == _FLIPPED:
            flipped += probability[r]
    return undetected, flipped


def _probabilities(values, ws, table, undetected, flipped):
    """Write each value's chance to flip no detector, and also an observable.

    A value whose signs were not walked gets -1 for both.
    """
    for i in numba.prange(len(values)):
        undetected[i], flipped[i] = _probability_of(values[i], ws, table)


probabilities = kernel(_probabilities)
"""``probabilities(work, values, ws, table, undetected, flipped)``."""


@jit
def _same_on(a, b, mask):
    """Return whether two rows agree on the bits of ``mask``."""
    same = True
    for k in range(len(mask)):
        if (a[k] ^ b[k]) & mask[k]:
            same = False
            break
    return same


@jit
def _sums_of(i, x, y, ws, y_signs, groups, table, rows, offset, fill):
    """Count, or write from ``rows[offset]`` on, the error sums of ``x[i]`` and ``y``.

    The deviations of a sign that agree on the detectors come one after the
    other in ``table``, so that each sum comes once. Returns their number, or
    -1 if one of their signs was not walked.
    """
    keys, key_index, key_start, order = groups
    sign_index, signs, start, deviations, _, detector_mask, observable_mask = table
    width = x.shape[1]
    key = np.empty(width, dtype=np.uint64)
    sign = np.empty(ws, dtype=np.uint64)
    found = 0
    for c in range(len(y_signs)):
        for k in range(ws):
            sign[k] = x[i, k] ^ y_signs[c, k]
            key[k] = y_signs[c, k]
        t = find(sign_index, signs, sign)
        if t < 0:
            return -1
        r = start[t]
        while r < start[t + 1]:
            # NOTE: rows r to e - 1 deviate alike on the detectors.
            e = r + 1
            while e < start[t + 1] and _same_on(
                deviations[e], deviations[r], detector_mask
            ):
                e += 1
            for k in range(width - ws):
                key[ws + k] = (x[i, ws + k] ^ deviations[r, k]) & detector_mask[k]
            g = find(key_index, keys, key)
            first, last = (key_start[g], key_start[g + 1]) if g >= 0 else (0, 0)
            for q in range(first, last):
                j = order[q]
                flipped = False
                for d in range(r, e):
                    for k in range(width - ws):
                        rest = x[i, ws + k] ^ y[j, ws + k] ^ deviations[d, k]
                        if rest & observable_mask[k]:
                            flipped = True
                            break
                    if flipped:
                        break
                if flipped:
                    if fill:
                        for k in range(width):
                            rows[offset + found, k] = x[i, k] ^ y[j, k]
                    found += 1
            r = e
    return found


def _accepted_sums(x, y, ws, y_signs, groups, table, counts, offsets, rows):
    """Count, or write, the sums of a value of ``x`` and one of ``y`` that are errors.

    ``groups`` is ``(keys, key_index, key_start, order)``: the values of ``y``
    with signs ``s`` and output flips on detectors ``d`` are the rows
    ``order[key_start[g]:key_start[g + 1]]``, where ``g`` is the row of
    ``keys`` that holds ``s`` and then ``d``. For each value of ``x``, each
    sign of ``y`` and each deviation of their sum, only the values of ``y``
    that cancel its detectors can be errors. ``counts[i]`` becomes the number
    of sums of ``x[i]``, or -1 if one of its signs was not walked, and with
    room in ``rows`` they go to the rows from ``offsets[i]`` on. ``x`` and
    ``y`` must be distinct, so that each sum comes once from each way of
    making it.
    """
    fill = len(rows) > 0
    for i in numba.prange(len(x)):
        counts[i] = _sums_of(
            i, x, y, ws, y_signs, groups, table, rows, offsets[i], fill
        )


accepted_sums = kernel(_accepted_sums)
"""``accepted_sums(work, x, y, ws, y_signs, groups, table, counts, offsets, rows)``."""


@jit
def _triples_of(i, a, b, c_index, c, rows, offset, fill):
    """Count, or write from ``rows[offset]`` on, the triples of ``a[i]``."""
    key = np.empty(a.shape[1], dtype=np.uint64)
    found = 0
    for j in range(len(b)):
        for k in range(a.shape[1]):
            key[k] = a[i, k] ^ b[j, k]
        t = find(c_index, c, key)
        if t >= 0:
            if fill:
                rows[offset + found, 0] = i
                rows[offset + found, 1] = j
                rows[offset + found, 2] = t
            found += 1
    return found


def _triples(a, b, c_index, c, counts, offsets, rows):
    """Count, or write, the ``(i, j, t)`` with ``a[i] ^ b[j] == c[t]``.

    ``counts[i]`` becomes the number of triples of ``a[i]``, and with room in
    ``rows`` they go to its rows from ``offsets[i]`` on.
    """
    fill = len(rows) > 0
    for i in numba.prange(len(a)):
        counts[i] = _triples_of(i, a, b, c_index, c, rows, offsets[i], fill)


triples = kernel(_triples)
"""``triples(work, a, b, c_index, c, counts, offsets, rows)``."""


@jit
def _unions_of(p, q, pieces, sets, rows, offset, fill):
    """Count, or write from ``rows[offset]`` on, the unions of two groups.

    ``p`` and ``q`` are the rows of ``pieces`` and of ``sets`` in the groups.
    """
    a, b = pieces.shape[1], sets.shape[1]
    union = np.empty(a + b, dtype=np.int32)
    found = 0
    for x in p:
        piece = pieces[x]
        for y in q:
            other = sets[y]
            if piece[0] >= other[0]:
                continue
            i = j = 0
            disjoint = True
            while i < a or j < b:
                if j == b or (i < a and piece[i] < other[j]):
                    union[i + j] = piece[i]
                    i += 1
                elif i == a or other[j] < piece[i]:
                    union[i + j] = other[j]
                    j += 1
                else:
                    disjoint = False
                    break
            if disjoint:
                if fill:
                    rows[offset + found] = union
                found += 1
    return found


def _join(
    pieces,
    piece_start,
    piece_order,
    sets,
    set_start,
    set_order,
    pairs,
    counts,
    offsets,
    rows,
):
    """Count, or write, the disjoint unions of a piece and a set, sorted.

    ``pairs[t]`` names a group of the rows of ``pieces`` and one of ``sets``,
    as ``group`` gives them, whose rows join. The rows of both are sorted. A
    union counts only if its piece holds its lowest fault: every union of
    pieces splits that way, and fewer splits repeat it. ``counts`` and
    ``offsets`` are as for ``_triples``.
    """
    fill = len(rows) > 0
    for t in numba.prange(len(pairs)):
        g, h = pairs[t, 0], pairs[t, 1]
        p = piece_order[piece_start[g] : piece_start[g + 1]]
        q = set_order[set_start[h] : set_start[h + 1]]
        counts[t] = _unions_of(p, q, pieces, sets, rows, offsets[t], fill)


join = kernel(_join)
"""``join(work, pieces, piece_start, piece_order, sets, set_start, set_order, pairs,
counts, offsets, rows)``."""
