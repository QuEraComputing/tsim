"""Numba kernels that join sets of faults whose direct detectors cancel.

A set of ``w`` faults, ascending, is a stored ``left`` set of the first
``w // 2`` and a streamed ``right`` set of the others. The left sets are
sorted by a GF(2)-linear hash of their direct detectors and then by their last
fault, so the left sets that complete a right set are a prefix of one run.
Each fault is a row of words: direct detectors ``D``, rotation signs ``S`` and
output flips ``O``; a set's words are the XOR of its faults' words.

``scan`` streams every right set. In ``COLLECT`` mode it gathers the distinct
``S`` of the sets whose direct detectors cancel, with one set of each; in
``EMIT`` mode it keeps the sets whose ``(S, O)`` the caller's table marks as a
possible logical error.
"""

from __future__ import annotations

import numba
import numpy as np

from tsim.utils.kernels import jit, kernel

COLLECT = 0
EMIT = 1

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


@jit
def _mix(x):
    """Return the splitmix64 finalizer of ``x``."""
    x = (x ^ (x >> np.uint64(30))) * _M1
    x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


@jit
def _words_hash(a, b, mask):
    """Return a hash of the words ``a`` followed by ``b & mask``."""
    h = _GOLDEN
    for k in range(len(a)):
        h = _mix(h ^ a[k])
    for k in range(len(b)):
        h = _mix(h ^ (b[k] & mask[k]))
    return h


@jit
def build_index(keys, capacity):
    """Return an open-addressing table of ``keys``, which must be distinct.

    Slot ``s`` holds the index of a key, or -1. ``capacity`` is a power of 2
    above ``len(keys)``.
    """
    table = np.full(capacity, -1, dtype=np.int64)
    mask = np.uint64(capacity - 1)
    for i in range(len(keys)):
        s = _mix(keys[i]) & mask
        while table[s] >= 0:
            s = (s + np.uint64(1)) & mask
        table[s] = i
    return table


@jit
def build_word_index(keys, masked, mask, capacity):
    """Return an open-addressing table of the rows ``(keys, masked & mask)``."""
    table = np.full(capacity, -1, dtype=np.int64)
    slots = np.uint64(capacity - 1)
    for i in range(len(keys)):
        s = _words_hash(keys[i], masked[i], mask) & slots
        while table[s] >= 0:
            s = (s + np.uint64(1)) & slots
        table[s] = i
    return table


@jit
def _find(table, keys, key):
    """Return the index of ``key`` in an index of ``build_index``, or -1."""
    mask = np.uint64(len(table) - 1)
    s = _mix(key) & mask
    while table[s] >= 0:
        if keys[table[s]] == key:
            return table[s]
        s = (s + np.uint64(1)) & mask
    return -1


@jit
def _find_words(table, keys, masked, a, b, mask):
    """Return the row of ``(a, b & mask)`` in a ``build_word_index``, or -1."""
    slots = np.uint64(len(table) - 1)
    s = _words_hash(a, b, mask) & slots
    while table[s] >= 0:
        i = table[s]
        same = True
        for k in range(len(a)):
            if keys[i, k] != a[k]:
                same = False
                break
        if same:
            for k in range(len(b)):
                if masked[i, k] != (b[k] & mask[k]):
                    same = False
                    break
        if same:
            return i
        s = (s + np.uint64(1)) & slots
    return -1


@jit
def _visit(mode, lane, h, c, r, faults, left, runs, signs, outputs, store, found):
    """Handle the right set ``c[:r]``, whose detector hash is ``h``.

    Returns False once the lane's buffer is full.
    """
    d_words, s_words, o_words, _ = faults
    left_sets, left_last, left_d, left_s, left_o = left
    run_keys, run_start, run_end, run_table = runs
    i = _find(run_table, run_keys, h)
    if i < 0:
        return True
    for e in range(run_start[i], run_end[i]):
        if left_last[e] >= c[0]:
            break
        cancels = True
        for k in range(d_words.shape[1]):
            word = left_d[e, k]
            for j in range(r):
                word ^= d_words[c[j], k]
            if word != 0:
                cancels = False
                break
        if not cancels:
            continue
        for k in range(s_words.shape[1]):
            word = left_s[e, k]
            for j in range(r):
                word ^= s_words[c[j], k]
            signs[k] = word
        for k in range(o_words.shape[1]):
            word = left_o[e, k]
            for j in range(r):
                word ^= o_words[c[j], k]
            outputs[k] = word
        if mode == COLLECT:
            if not _collect(lane, e, c, r, signs, left_sets, store):
                return False
        elif not _emit(lane, e, c, r, signs, outputs, left_sets, found):
            return False
    return True


@jit
def _collect(lane, e, c, r, signs, left_sets, store):
    """Add the signs of a set to the lane's set of distinct signs."""
    keys, table, sets, count = store
    capacity = keys.shape[1]
    slots = np.uint64(table.shape[1] - 1)
    s = _words_hash(signs, signs[:0], signs[:0]) & slots
    while table[lane, s] >= 0:
        i = table[lane, s]
        same = True
        for k in range(len(signs)):
            if keys[lane, i, k] != signs[k]:
                same = False
                break
        if same:
            return True
        s = (s + np.uint64(1)) & slots
    n = count[lane]
    if n == capacity:
        return False
    table[lane, s] = n
    keys[lane, n] = signs
    _write_set(sets[lane, n], e, c, r, left_sets)
    count[lane] = n + 1
    return True


@jit
def _emit(lane, e, c, r, signs, outputs, left_sets, found):
    """Keep a set whose signs and output flips can flip an observable."""
    keys, masked, det_mask, obs_mask, multi, obs, table, sets, count = found
    i = _find_words(table, keys, masked, signs, outputs, det_mask)
    if i < 0:
        return True
    if not multi[i]:
        same = True
        for k in range(len(outputs)):
            if (outputs[k] & obs_mask[k]) != obs[i, k]:
                same = False
                break
        if same:
            return True
    n = count[lane]
    if n == sets.shape[1]:
        return False
    _write_set(sets[lane, n], e, c, r, left_sets)
    count[lane] = n + 1
    return True


@jit
def _write_set(out, e, c, r, left_sets):
    """Write the left set ``e`` followed by ``c[:r]``."""
    size = left_sets.shape[1]
    for j in range(size):
        out[j] = left_sets[e, j]
    for j in range(r):
        out[size + j] = c[j]


def _scan(mode, r, faults, left, runs, store, found):
    """Stream every right set of ``r`` faults through ``_visit``.

    Lane ``lane`` takes the right sets whose first fault is ``lane`` modulo the
    number of lanes. Returns whether a lane's buffer filled up.
    """
    _, s_words, o_words, hashes = faults
    n = len(hashes)
    lanes = store[3].shape[0]
    full = np.zeros(lanes, dtype=np.bool_)
    for lane in numba.prange(lanes):
        c = np.zeros(3, dtype=np.int64)
        signs = np.zeros(s_words.shape[1], dtype=np.uint64)
        outputs = np.zeros(o_words.shape[1], dtype=np.uint64)
        for c0 in range(lane, n, lanes):
            if full[lane]:
                break
            c[0] = c0
            h0 = hashes[c0]
            if r == 1:
                if not _visit(
                    mode,
                    lane,
                    h0,
                    c,
                    r,
                    faults,
                    left,
                    runs,
                    signs,
                    outputs,
                    store,
                    found,
                ):
                    full[lane] = True
                continue
            for c1 in range(c0 + 1, n):
                c[1] = c1
                h1 = h0 ^ hashes[c1]
                if r == 2:
                    if not _visit(
                        mode,
                        lane,
                        h1,
                        c,
                        r,
                        faults,
                        left,
                        runs,
                        signs,
                        outputs,
                        store,
                        found,
                    ):
                        full[lane] = True
                        break
                    continue
                for c2 in range(c1 + 1, n):
                    c[2] = c2
                    if not _visit(
                        mode,
                        lane,
                        h1 ^ hashes[c2],
                        c,
                        r,
                        faults,
                        left,
                        runs,
                        signs,
                        outputs,
                        store,
                        found,
                    ):
                        full[lane] = True
                        break
                if full[lane]:
                    break
    return full.any()


scan = kernel(_scan)
"""``scan(work, mode, r, faults, left, runs, store, found)``, in parallel when
``work`` (the number of right sets) is large."""
