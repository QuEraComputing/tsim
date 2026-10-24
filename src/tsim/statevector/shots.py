"""The fused shot kernel: noise drawn straight into each shot's packed state.

A shot's noise reaches its outputs only through parities of its f-bits: each
direct output is one f-bit, and a statevector component reads its packed f-bits
(lanes), its key and the f-part of its outputs (tables and trees). Folding each
fault outcome's f-bits into its signature on these words, at compile time, lets
a block's faults XOR straight into a few words per shot: no f matrix is built,
and the postselection screen tests a word. Each row then writes its direct
outputs in column order, and its survivors' words feed the components.

The faults are drawn exactly as ``ChannelSampler`` draws them, block by block
from the same streams, so the noise is the same as with an f matrix.
"""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np

from tsim.noise.channels import FireTables, _block_faults
from tsim.utils.kernels import jit, kernel


class ComponentWords(NamedTuple):
    """Where a component's words sit in a shot's state.

    Attributes:
        symbols: First word of its packed f-bits, as lanes read them.
        symbol_words: Their 64-bit words.
        key: First word of its key, -1 without tables or trees.
        key_words: The key's words.
        pattern: First word of the f-part of its outputs.
        pattern_words: Its words.

    """

    symbols: int
    symbol_words: int
    key: int
    key_words: int
    pattern: int
    pattern_words: int


class ShotLayout(NamedTuple):
    """A shot's packed state and how faults change it, for one screen.

    Attributes:
        words: 64-bit words per shot.
        num_direct: Direct outputs, bits ``0 … num_direct-1`` of the state.
        components: Each statevector component's words.
        signature_start: First signature entry of each fault outcome, shape
            ``(K + 1,)``.
        signature_word: The word of each entry, shape ``(S,)``.
        signature_bits: The bits it flips, shape ``(S,)``.
        flips: Each direct output's flip, with its reference bit, packed, shape
            ``(words,)``; zero beyond the direct outputs.
        screens: The postselected direct detectors, packed, shape ``(words,)``.
        direct_bit: The direct output of each output column, -1 for a
            component's, shape ``(num_outputs,)``.
        kept: The output columns a discarded shot keeps: its direct detectors,
            shape ``(num_outputs,)``.

    """

    words: int
    num_direct: int
    components: tuple[ComponentWords | None, ...]
    signature_start: np.ndarray
    signature_word: np.ndarray
    signature_bits: np.ndarray
    flips: np.ndarray
    screens: np.ndarray
    direct_bit: np.ndarray
    kept: np.ndarray


def _packed_bits(bits: np.ndarray, words: int) -> np.ndarray:
    """Return booleans as 64-bit words, bit ``i`` in word ``i // 64``."""
    out = np.zeros(words, dtype=np.uint64)
    for i in np.flatnonzero(bits):
        out[i >> 6] |= np.uint64(1) << np.uint64(i & 63)
    return out


def shot_layout(
    noise: FireTables,
    num_f: int,
    direct_f_indices: np.ndarray,
    direct_flips: np.ndarray,
    direct_columns: np.ndarray,
    num_outputs: int,
    num_detectors: int,
    mask: np.ndarray | None,
    components: list[tuple[np.ndarray, int, np.ndarray | None, np.ndarray | None]],
) -> ShotLayout:
    """Return the shot layout of a program under a postselection mask.

    Args:
        noise: The noise model's sampling tables.
        num_f: The number of f-bits.
        direct_f_indices: The f-bit of each direct output, -1 for a constant.
        direct_flips: Each direct output's flip, with its reference bit.
        direct_columns: The output column of each direct output.
        num_outputs: The number of output columns.
        num_detectors: The detector columns come first.
        mask: The postselected detectors, if any.
        components: Per statevector component, its f-bits (``f_selection``), its
            lanes' 32-bit words, and for tables or trees the key bits and output
            pattern bits each of its f-bits flips (64-bit words), or None.

    """
    num_direct = len(direct_f_indices)
    words = (num_direct + 63) // 64
    placed: list[ComponentWords | None] = []
    for _, num_words, key_columns, pattern_columns in components:
        symbol_words = (num_words + 1) // 2
        symbols, words = words, words + symbol_words
        if key_columns is None or pattern_columns is None:
            placed.append(ComponentWords(symbols, symbol_words, -1, 0, -1, 0))
            continue
        key, words = words, words + key_columns.shape[1]
        pattern, words = words, words + pattern_columns.shape[1]
        placed.append(
            ComponentWords(
                symbols,
                symbol_words,
                key,
                key_columns.shape[1],
                pattern,
                pattern_columns.shape[1],
            )
        )
    words = max(words, 1)
    contribution = np.zeros((num_f, words), dtype=np.uint64)
    for j, f in enumerate(direct_f_indices):
        if f >= 0:
            contribution[f, j >> 6] ^= np.uint64(1) << np.uint64(j & 63)
    for (selection, _, key_columns, pattern_columns), where in zip(
        components, placed, strict=True
    ):
        assert where is not None
        for i, f in enumerate(selection):
            word = where.symbols + (i >> 6)
            contribution[f, word] ^= np.uint64(1) << np.uint64(i & 63)
            if key_columns is not None and pattern_columns is not None:
                key = slice(where.key, where.key + where.key_words)
                contribution[f, key] ^= key_columns[i]
                pattern = slice(where.pattern, where.pattern + where.pattern_words)
                contribution[f, pattern] ^= pattern_columns[i]
    start, word, bits = _signatures(noise.flip_start, noise.flip_cols, contribution)
    flips = _packed_bits(np.asarray(direct_flips, dtype=bool), words)
    screened = np.zeros(num_direct, dtype=bool)
    is_detector = direct_columns < num_detectors
    if mask is not None:
        screened[is_detector] = mask[direct_columns[is_detector]]
    direct_bit = np.full(num_outputs, -1, dtype=np.int64)
    direct_bit[direct_columns] = np.arange(num_direct)
    kept = np.zeros(num_outputs, dtype=np.bool_)
    kept[direct_columns[is_detector]] = True
    return ShotLayout(
        words=words,
        num_direct=num_direct,
        components=tuple(placed),
        signature_start=start,
        signature_word=word,
        signature_bits=bits,
        flips=flips,
        screens=_packed_bits(screened, words),
        direct_bit=direct_bit,
        kept=kept,
    )


@jit
def _signatures(flip_start, flip_cols, contribution):
    """Return each fault outcome's signature: the words its f-bits flip."""
    outcomes, words = len(flip_start) - 1, contribution.shape[1]
    start = np.zeros(outcomes + 1, dtype=np.int64)
    row = np.zeros(words, dtype=np.uint64)
    entries = 0
    for k in range(outcomes):
        row[:] = 0
        for j in range(flip_start[k], flip_start[k + 1]):
            for w in range(words):
                row[w] ^= contribution[flip_cols[j], w]
        for w in range(words):
            entries += row[w] != 0
        start[k + 1] = entries
    word = np.empty(entries, dtype=np.int64)
    bits = np.empty(entries, dtype=np.uint64)
    e = 0
    for k in range(outcomes):
        row[:] = 0
        for j in range(flip_start[k], flip_start[k + 1]):
            for w in range(words):
                row[w] ^= contribution[flip_cols[j], w]
        for w in range(words):
            if row[w] != 0:
                word[e], bits[e] = w, row[w]
                e += 1
    return start, word, bits


@jit
def _sample_block(
    out,
    state,
    discarded,
    seed,
    lo,
    hi,
    p_fire,
    outcome_start,
    cdf,
    signature_start,
    signature_word,
    signature_bits,
    flips,
    screens,
    direct_bit,
    kept,
):
    """Draw the noise of rows ``lo … hi-1`` from the stream ``seed``."""
    words = state.shape[1]
    expected = (hi - lo) * np.sum(p_fire)
    capacity = int(expected + 8 * np.sqrt(expected)) + 64
    while True:
        fault_row = np.empty(capacity, dtype=np.int64)
        fault_outcome = np.empty(capacity, dtype=np.int64)
        count = _block_faults(
            hi - lo, seed, p_fire, outcome_start, cdf, fault_row, fault_outcome
        )
        if count >= 0:
            break
        capacity *= 2
    for r in range(lo, hi):
        for w in range(words):
            state[r, w] = flips[w]
    for f in range(count):
        r = lo + fault_row[f]
        k = fault_outcome[f]
        for e in range(signature_start[k], signature_start[k + 1]):
            state[r, signature_word[e]] ^= signature_bits[e]
    for r in range(lo, hi):
        discard = np.uint64(0)
        for w in range(words):
            discard |= state[r, w] & screens[w]
        discarded[r] = discard != 0
        row = out[r]
        for c in range(len(direct_bit)):
            j = direct_bit[c]
            if j < 0 or (discard and not kept[c]):
                row[c] = False
            else:
                row[c] = (state[r, j >> 6] >> np.uint64(j & 63)) & np.uint64(1)


@kernel
def sample_shots(
    out,
    state,
    seeds,
    block_shots,
    p_fire,
    outcome_start,
    cdf,
    signature_start,
    signature_word,
    signature_bits,
    flips,
    screens,
    direct_bit,
    kept,
):
    """Draw the noise of ``out``'s rows into ``state`` and write their direct outputs.

    Row ``r`` is shot ``r`` of block ``r // block_shots``, drawn from the stream
    ``seeds[r // block_shots]`` as ``ChannelSampler`` draws it; the blocks run
    in parallel. A shot that a postselected direct detector discards keeps its
    direct detectors and reads False elsewhere; the others read their direct
    outputs and False in the components' columns.

    Returns:
        Which rows are discarded.

    """
    rows = out.shape[0]
    discarded = np.zeros(rows, dtype=np.bool_)
    for b in numba.prange(len(seeds)):
        lo = b * block_shots
        _sample_block(
            out,
            state,
            discarded,
            seeds[b],
            lo,
            min(lo + block_shots, rows),
            p_fire,
            outcome_start,
            cdf,
            signature_start,
            signature_word,
            signature_bits,
            flips,
            screens,
            direct_bit,
            kept,
        )
    return discarded


@jit
def gather(state, rows, first, count):
    """Return words ``first … first+count-1`` of the given rows of ``state``."""
    out = np.empty((len(rows), count), dtype=np.uint64)
    for i in range(len(rows)):
        for w in range(count):
            out[i, w] = state[rows[i], first + w]
    return out


@jit
def gather_symbols(state, rows, first, num_words):
    """Return the packed f-bits at word ``first`` of the given rows, as lanes read them."""
    out = np.empty((len(rows), num_words), dtype=np.uint32)
    for i in range(len(rows)):
        for w in range(num_words):
            word = state[rows[i], first + (w >> 1)]
            out[i, w] = np.uint32(
                (word >> np.uint64(32 * (w & 1))) & np.uint64(0xFFFFFFFF)
            )
    return out


@jit
def scatter(out, rows, columns, samples):
    """Write ``samples[i, j]`` to ``out[rows[i], columns[j]]``."""
    for i in range(len(rows)):
        for j in range(len(columns)):
            out[rows[i], columns[j]] = samples[i, j]
