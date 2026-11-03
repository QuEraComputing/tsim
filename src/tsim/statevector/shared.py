"""Trajectory-sharing executor for statevector components, on the CPU with numba.

A shot reaches a component's amplitudes only through its rotation signs, which
read parities of its f-bits (the component's *key*) and the outcomes of earlier
measurements. A postselected output prunes a shot through its f-part and its
outcomes. The noise sampler writes each shot's key and output f-parts into the
shot's words (``tsim.statevector.shots``), so this executor never sees f-bits.

Lanes. The shots of a chunk with equal keys and equal f-parts of the
postselected outputs share a lane: its amplitudes and its *lane symbols*, the key
bits, then those f-parts, then the branch bits of the lane's outcomes.
``lane_program`` rewrites the rotation signs, the pruning tests and the outputs'
branch parts over lane symbols.

Outcomes. At a measurement every live shot compares its own random number, drawn
from a counter-based stream by its index and the action, with its lane's chance
of outcome 1. A lane splits into one lane per outcome its shots drew. A shot's
outcomes so depend only on the seed, its index and its key: not on the other
shots, the chunk, the threads or whether lanes are shared at all.

Pruning. After the action that sets a postselected output's last branch bit (or
before the first action, for an output of f-bits only) the lanes in which it reads
1 die. Their shots keep the outcomes drawn so far, and their outputs that need a
later branch bit read False.

Outputs. Each shot XORs the branch part of its outputs, from its last lane, and
its own fair coins: the dormant measurements that no sign and no postselected
output reads, drawn per shot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, cast

import numba
import numpy as np
from numba import types
from numba.extending import intrinsic

from tsim.statevector.program import (
    MEASURE,
    MEASURE_DIAG,
    MEASURE_DORMANT,
    ROTATE,
    ROTATE_DIAG,
    StatevectorComponent,
)
from tsim.utils.kernels import jit, kernel
from tsim.utils.linalg import find_basis
from tsim.utils.rng import GOLDEN, splitmix64

_TINY = 1e-30
_I_POWERS = (1, 1j, -1, -1j)
# NOTE: the largest uint64 a float64 below 2^64 converts to.
_MAX_THRESHOLD = np.uint64(0xFFFF_FFFF_FFFF_F800)


class LaneProgram(NamedTuple):
    """A component's tape over lane symbols, for one postselection mask.

    Lane symbols are ``lane_words`` uint64 words: the key's words, then the shot
    words that hold postselected outputs, masked to them (so they hold their
    f-parts), then the branch bits of the measurements that split lanes.

    Attributes:
        width: Number of qubits of a lane's amplitudes.
        kind: Kernel of each action, ``(A,)``.
        extent: Each action's active qubits: it touches the first
            ``2^extent`` amplitudes of a lane, ``(A,)``.
        x_mask: X part of each action's Pauli, ``(A,)``.
        z_mask: Z part of each action's Pauli, ``(A,)``.
        pivot: Qubit that each measurement frees, ``(A,)``.
        cos: Each rotation's ``cos θ``, ``(A,)``.
        key_start: First word of the component's key in the shot words.
        key_words: Number of words of the key.
        ps_word: The shot words that hold postselected outputs, ``(W,)``.
        ps_mask: The postselected outputs' bits in them, ``(W,)``.
        lane_words: Number of words of lane symbols.
        splits: Whether each action is a measurement that splits lanes, ``(A,)``.
        branch: Lane bit that each action sets, -1 for none, ``(A,)``.
        sign_row: Lane bits of each rotation's sign, ``(A, lane_words)``.
        sign_const: Constant of each rotation's sign, ``(A,)``.
        prune_first: Postselected outputs of f-bits only, ``(O,)``.
        pruned: Actions after which postselected outputs are tested, ``(A,)``.
        prune_outputs: The outputs tested after each action, ``(A, O)``.
        prune_row: Lane bits of each postselected output, ``(O, lane_words)``: its
            f-part bit, which holds its constant and reference bit, and its branches.
        out_words: The shot words that hold the component's outputs, ``(W,)``.
        output_word: Each output's word in ``out_words``, ``(O,)``.
        output_bit: Each output's bit in its word, ``(O,)``.
        branch_row: Lane bits of each output's branch part, ``(O, lane_words)``.
        ready: Action that sets each output's last branch bit, -1 for none, ``(O,)``.
        coin_pattern: Output bits that each fair coin flips in ``out_words``,
            ``(C, W)``.
        i_power: ``i^|x∧z|`` of each action's Pauli ``X^x Z^z``, ``(A,)``.
        factor: Each action's diagonal rotation factors ``cos θ ∓ i sin θ``,
            ``(A, 2)``.
        coupling: Each action's rotation coupling ``-i sin θ i^|x∧z|``, ``(A,)``.

    """

    width: int
    kind: np.ndarray
    extent: np.ndarray
    x_mask: np.ndarray
    z_mask: np.ndarray
    pivot: np.ndarray
    cos: np.ndarray
    key_start: int
    key_words: int
    ps_word: np.ndarray
    ps_mask: np.ndarray
    lane_words: int
    splits: np.ndarray
    branch: np.ndarray
    sign_row: np.ndarray
    sign_const: np.ndarray
    prune_first: np.ndarray
    pruned: np.ndarray
    prune_outputs: np.ndarray
    prune_row: np.ndarray
    out_words: np.ndarray
    output_word: np.ndarray
    output_bit: np.ndarray
    branch_row: np.ndarray
    ready: np.ndarray
    coin_pattern: np.ndarray
    i_power: np.ndarray
    factor: np.ndarray
    coupling: np.ndarray


class KeyBasis(NamedTuple):
    """A component's key: parities of its f-bits that its rotation signs read.

    Attributes:
        rows: Key bit ``k`` is the parity of the component f-bits in ``rows[k]``,
            shape ``(K, num_f)``.
        sign_coords: Each action's sign f-part in key bits, ``(A, K)``: the sign
            reads the parity of the key bits it marks.

    """

    rows: np.ndarray
    sign_coords: np.ndarray


def unpack_symbols(words: np.ndarray, num_words: int) -> np.ndarray:
    """Return uint32 symbol words as booleans: symbol s is bit s % 32 of word s // 32."""
    shifts = np.arange(32, dtype=np.uint32)
    bits = (words[..., None] >> shifts) & 1
    return bits.reshape(*words.shape[:-1], 32 * num_words).astype(bool)


def pack_bits(bits: np.ndarray, num_words: int) -> np.ndarray:
    """Return booleans as uint64 words: bit i is bit i % 64 of word i // 64."""
    padded = np.zeros((*bits.shape[:-1], 64 * num_words), dtype=bool)
    padded[..., : bits.shape[-1]] = bits
    return np.packbits(padded, axis=-1, bitorder="little").view("<u8").astype(np.uint64)


def key_basis(component: StatevectorComponent) -> KeyBasis:
    """Return a basis of the f-parts of the component's rotation signs."""
    tape = component.tape
    num_f = len(tape.f_selection)
    signs = unpack_symbols(tape.sign_row, component.num_words)[:, :num_f]
    rotation = np.isin(tape.kind, (ROTATE_DIAG, ROTATE))
    signs = signs & rotation[:, None]
    nonzero = signs.any(axis=1)
    if not nonzero.any():
        return KeyBasis(np.zeros((0, num_f), bool), np.zeros((len(signs), 0), bool))
    rows, _ = find_basis(signs[nonzero].astype(np.uint8))
    rows = np.asarray(rows, dtype=bool)
    # NOTE: the basis rows come first, so each sign's coordinates are its transform.
    _, transform = find_basis(np.vstack([rows, signs]).astype(np.uint8))
    coords = np.asarray(transform, dtype=bool)[len(rows) :]
    return KeyBasis(rows, coords)


def lane_program(
    component: StatevectorComponent,
    key: KeyBasis,
    key_start: int,
    postselected: np.ndarray,
) -> LaneProgram:
    """Return the component's tape over lane symbols.

    Args:
        component: The compiled component.
        key: Its key basis.
        key_start: First word of the key in the shot words.
        postselected: Whether each output slot is postselected, ``(O,)``.

    """
    tape = component.tape
    num_actions = len(tape.kind)
    num_outputs = component.num_outputs
    kind, branch = tape.kind, tape.branch
    symbol_bits = 32 * component.num_words
    sign_bits = unpack_symbols(tape.sign_row, component.num_words)
    output_bits = unpack_symbols(tape.output_row, component.num_words)
    postselected = np.asarray(postselected, dtype=bool)

    setter = np.full(symbol_bits, -1, dtype=np.int64)
    setter[branch[branch >= 0]] = np.flatnonzero(branch >= 0)
    ready = np.max(np.where(output_bits, setter[None, :], -1), axis=1, initial=-1)

    # NOTE: a dormant measurement splits lanes when a sign or a postselected output
    # reads its branch; otherwise it is a fair coin per shot.
    rotation = np.isin(kind, (ROTATE_DIAG, ROTATE))
    read = np.asarray(sign_bits[rotation].any(axis=0)) | np.asarray(
        output_bits[postselected].any(axis=0)
    )
    dormant = kind == MEASURE_DORMANT
    lane_dormant = dormant & read[branch]
    splits = np.isin(kind, (MEASURE_DIAG, MEASURE)) | lane_dormant
    coin = dormant & ~lane_dormant

    num_key = len(key.rows)
    key_words = -(-num_key // 64)
    output_index = np.asarray(component.output_indices, dtype=np.int64)
    ps_word, ps_index = np.unique(output_index[postselected] >> 6, return_inverse=True)
    ps_mask = np.zeros(len(ps_word), dtype=np.uint64)
    np.bitwise_or.at(
        ps_mask,
        ps_index,
        np.uint64(1) << (output_index[postselected] & 63).astype(np.uint64),
    )
    first_branch = 64 * (key_words + len(ps_word))
    lane_words = max(1, -(-(first_branch + int(splits.sum())) // 64))
    # NOTE: the lane bit of each symbol: f-bits have none, lane-splitting branches do.
    lane_bit = np.full(symbol_bits, -1, dtype=np.int64)
    lane_bit[branch[splits]] = first_branch + np.arange(int(splits.sum()))

    def branch_part(rows: np.ndarray) -> np.ndarray:
        """Lane bits of the lane-splitting branches in symbol rows."""
        out = np.zeros((len(rows), 64 * lane_words), dtype=bool)
        carried = lane_bit >= 0
        for i, row in enumerate(rows):
            out[i, lane_bit[row & carried]] = True
        return out

    signs = branch_part(sign_bits)
    signs[:, :num_key] = key.sign_coords
    signs[~rotation] = False
    prunes = branch_part(output_bits)
    slots = np.flatnonzero(postselected)
    prunes[slots, 64 * (key_words + ps_index) + (output_index[slots] & 63)] = True
    prunes[~postselected] = False

    prune_outputs = np.zeros((num_actions, num_outputs), dtype=bool)
    slots = np.flatnonzero(postselected & (ready >= 0))
    prune_outputs[ready[slots], slots] = True

    out_words, output_word = np.unique(output_index >> 6, return_inverse=True)
    local_bit = 64 * output_word + (output_index & 63)
    coin_symbols = branch[coin]
    coin_pattern = np.zeros((len(coin_symbols), 64 * len(out_words)), dtype=bool)
    for j, symbol in enumerate(coin_symbols):
        coin_pattern[j, local_bit[output_bits[:, symbol]]] = True

    quarter_turns = [
        int(x & z).bit_count() & 3
        for x, z in zip(tape.x_mask, tape.z_mask, strict=True)
    ]
    i_power = np.array([_I_POWERS[q] for q in quarter_turns], dtype=np.complex128)
    cos, sin = tape.cos, tape.sin
    coupling = -1j * sin * i_power
    # NOTE: _rotate_range mixes pairs with a real or an imaginary coupling alone.
    assert np.all((coupling.real == 0) | (coupling.imag == 0))
    lane_branch = np.where(splits, lane_bit[np.maximum(branch, 0)], -1)
    return LaneProgram(
        width=component.width,
        kind=tape.kind,
        extent=tape.extent,
        x_mask=tape.x_mask,
        z_mask=tape.z_mask,
        pivot=tape.pivot,
        cos=cos,
        key_start=key_start,
        key_words=key_words,
        ps_word=ps_word.astype(np.int64),
        ps_mask=ps_mask,
        lane_words=lane_words,
        splits=splits,
        branch=lane_branch.astype(np.int64),
        sign_row=pack_bits(signs, lane_words),
        sign_const=tape.sign_const.astype(np.int64),
        prune_first=postselected & (ready < 0),
        pruned=np.asarray(prune_outputs.any(axis=1)),
        prune_outputs=prune_outputs,
        prune_row=pack_bits(prunes, lane_words),
        out_words=out_words.astype(np.int64),
        output_word=output_word.astype(np.int64),
        output_bit=output_index & 63,
        branch_row=pack_bits(branch_part(output_bits), lane_words),
        ready=ready.astype(np.int64),
        coin_pattern=pack_bits(coin_pattern, len(out_words)),
        i_power=i_power,
        factor=np.stack([cos - 1j * sin, cos + 1j * sin], axis=1),
        coupling=coupling,
    )


def max_lane_words(component: StatevectorComponent, key: KeyBasis) -> int:
    """Return the most words of lane symbols that ``lane_program`` gives a component.

    That is under any mask: the key's words, every word of its outputs, and a bit
    per measurement.
    """
    tape = component.tape
    output_words = np.unique(np.asarray(component.output_indices, dtype=np.int64) >> 6)
    measurements = int(
        np.isin(tape.kind, (MEASURE_DIAG, MEASURE, MEASURE_DORMANT)).sum()
    )
    bits = 64 * (-(-len(key.rows) // 64) + len(output_words)) + measurements
    return max(1, -(-bits // 64))


def lane_bytes(component: StatevectorComponent, key: KeyBasis) -> int:
    """Return the peak memory of one lane of ``run_lanes``.

    That is three rows of amplitudes (the old and the new array while lanes grow),
    four rows of lane symbols (its own, and a grave's while the graves grow) and
    64 bytes of per-lane scratch.
    """
    amplitudes = 3 * 2**component.width * np.dtype(np.complex128).itemsize
    return amplitudes + 4 * 8 * max_lane_words(component, key) + 64


@intrinsic
def _ctpop(typingctx, value):
    def codegen(context, builder, signature, args):
        return builder.ctpop(args[0])

    return types.uint64(types.uint64), codegen


# NOTE: the cast only tells type checkers how the kernels call it.
_popcount = cast(Callable[[np.uint64], np.uint64], _ctpop)


@jit
def _parity(value):
    """Return the parity of the bits of a uint64."""
    return np.int64(_popcount(value) & np.uint64(1))


@jit
def _insert_zero(value, bit):
    """Return ``value`` with a 0 bit inserted at the position of the power of 2 ``bit``."""
    return (value & -bit) << 1 | (value & (bit - 1))


@jit
def _row_parity(symbols, row, mask):
    """Return the parity of ``symbols[row] & mask``."""
    acc = np.uint64(0)
    for w in range(len(mask)):
        acc ^= symbols[row, w] & mask[w]
    return _parity(acc)


@jit
def _copy_lane(amps, target, source, span):
    """Copy the first ``span`` amplitudes of lane ``source`` to lane ``target``."""
    for i in range(span):
        amps[target, i] = amps[source, i]


@jit
def _clear(amps, lanes, lo, hi):
    """Set amplitudes ``lo … hi-1`` of the first ``lanes`` lanes to 0."""
    for lane in range(lanes):
        for i in range(lo, hi):
            amps[lane, i] = 0


@jit
def _grown(array, rows, used):
    """Return ``array`` with ``rows`` rows, keeping the first ``used``."""
    out = np.zeros((rows, array.shape[1]), dtype=array.dtype)
    out[:used] = array[:used]
    return out


@jit
def _rotate_range(amps, lo, hi, span, flip, diagonal, x, z, cos, factor, coupling):
    """Rotate the units ``lo … hi-1`` of the lanes' first ``span`` amplitudes.

    A diagonal rotation's units are the amplitudes i, which it multiplies by
    ``factor[sign ^ parity(i & z)]``. The others' are the pairs of amplitudes i
    and ``j = i ^ x``, which they mix with ``cos`` and ``±coupling``. A lane's
    units follow the previous lane's.
    """
    if diagonal:
        for lane in range(lo // span, -(-hi // span)):
            for i in range(max(lo - lane * span, 0), min(hi - lane * span, span)):
                amps[lane, i] *= factor[flip[lane] ^ _parity(np.uint64(i & z))]
        return
    # NOTE: parity(j & z) is parity(i & z) ^ parity(x & z), and the coupling is
    # real or imaginary, so a pair takes one parity and real products.
    values = amps.view(np.float64)
    pairs = span >> 1
    low = x & -x
    imaginary = coupling.real == 0
    strength = coupling.imag if imaginary else coupling.real
    partner = -1.0 if _parity(np.uint64(x & z)) else 1.0
    for lane in range(lo // pairs, -(-hi // pairs)):
        row = values[lane]
        lane_strength = -strength if flip[lane] else strength
        for t in range(max(lo - lane * pairs, 0), min(hi - lane * pairs, pairs)):
            i = _insert_zero(t, low)
            j = i ^ x
            si = -lane_strength if _parity(np.uint64(i & z)) else lane_strength
            sj = partner * si
            pr, pi, qr, qi = row[2 * i], row[2 * i + 1], row[2 * j], row[2 * j + 1]
            if imaginary:
                row[2 * i], row[2 * i + 1] = cos * pr - sj * qi, cos * pi + sj * qr
                row[2 * j], row[2 * j + 1] = cos * qr - si * pi, cos * qi + si * pr
            else:
                row[2 * i], row[2 * i + 1] = cos * pr + sj * qr, cos * pi + sj * qi
                row[2 * j], row[2 * j + 1] = cos * qr + si * pr, cos * qi + si * pi


@jit
def _weights_range(amps, lo, hi, span, x, z, i_power, w0, w1):
    """Add the outcome weights of measuring ``X^x Z^z`` in lanes ``lo … hi-1``."""
    for lane in range(lo, hi):
        zero, one = w0[lane], w1[lane]
        for i in range(span):
            a = amps[lane, i]
            if x == 0:
                if _parity(np.uint64(i & z)):
                    one += a.real**2 + a.imag**2
                else:
                    zero += a.real**2 + a.imag**2
                continue
            b = i_power * amps[lane, i ^ x]
            if _parity(np.uint64((i ^ x) & z)):
                b = -b
            plus, minus = a + b, a - b
            zero += (plus.real**2 + plus.imag**2) / 4
            one += (minus.real**2 + minus.imag**2) / 4
        w0[lane], w1[lane] = zero, one


@jit
def _project_range(amps, lo, hi, span, x, z, pivot, i_power, outcome, weight):
    """Project lanes ``lo … hi-1`` onto their outcome and free the pivot.

    The pivot-1 half of the amplitudes becomes 0: the planner moved the measured
    Pauli onto the pivot, so the pivot ends in ``|0⟩``. The pivot is a bit of
    ``x`` when ``x`` is nonzero, else of ``z``: the loop pairs each pivot-0
    amplitude i with ``i ^ x``, which must then lie in the pivot-1 half.
    """
    bit = 1 << pivot
    for lane in range(lo, hi):
        if x == 0:
            scale = 1 / np.sqrt(max(weight[lane], _TINY))
        else:
            scale = 1 / np.sqrt(max(2 * weight[lane], _TINY))
        for t in range(span >> 1):
            i = _insert_zero(t, bit)
            if x == 0:
                j = i | bit
                keep = _parity(np.uint64(i & z)) == outcome[lane]
                amps[lane, i] = (amps[lane, i] if keep else amps[lane, j]) * scale
            else:
                j = i ^ x
                b = i_power * amps[lane, j]
                if _parity(np.uint64(j & z)):
                    b = -b
                if outcome[lane]:
                    b = -b
                amps[lane, i] = (amps[lane, i] + b) * scale
            amps[lane, j] = 0


@jit
def _chunks(work, units, parallel_work, threads):
    """Return how many chunks a loop over ``units`` runs in: 1 below ``parallel_work``."""
    return 1 if work < parallel_work else max(1, min(units, 4 * threads))


@jit
def _poison(amps, size, span):
    """Fill every amplitude the executor must not read with NaN (for tests)."""
    amps[:size, span:] = np.nan
    amps[size:] = np.nan


@jit
def _threshold(p1):
    """Return the uint64 below which a draw is outcome 1, for ``0 < p1 < 1``.

    A uniform 64-bit draw is below it with probability ``p1`` to within 2^-64.
    """
    scaled = p1 * 18446744073709551616.0
    return np.uint64(scaled) if scaled < 1.8446744073709550e19 else _MAX_THRESHOLD


@jit
def _cut(w0, w1):
    """Return ``(certain, threshold)`` for a measurement of outcome weights w0, w1.

    ``certain`` is the outcome if it is certain, else -1, and then a uniform 64-bit
    draw below ``threshold`` is outcome 1. The threshold comes from the rarer
    outcome's chance, so that a rare 0 keeps its chance as a rare 1 does: when
    ``w0`` is the smaller, the draws from ``2^64 - _threshold(p0)`` up are 0.
    """
    total = w0 + w1
    if not total > _TINY:
        return 0, np.uint64(0)
    if w1 <= w0:
        rare = _threshold(w1 / total) if w1 > 0.0 else np.uint64(0)
        return (0 if rare == 0 else -1), rare
    rare = _threshold(w0 / total) if w0 > 0.0 else np.uint64(0)
    return (1 if rare == 0 else -1), np.uint64(0) - rare


@jit
def _draw(seed, shot, slots, slot):
    """Return shot ``shot``'s random number in draw slot ``slot`` of the stream ``seed``."""
    index = np.uint64(shot) * np.uint64(slots) + np.uint64(slot)
    _, value = splitmix64(seed + index * GOLDEN)
    return value


# NOTE: _lane_key runs once per shot and takes only the LaneProgram fields it
# reads. Called from a kernel loaded from numba's disk cache it is not inlined,
# and passing the whole LaneProgram (~200 machine words) by value made
# trajectory sharing 5-8x slower on x86.
@jit
def _lane_key(words, shot, key_start, key_words, ps_word, ps_mask, out):
    """Write a shot's first lane symbols, its key and postselected f-parts, to ``out``."""
    for w in range(key_words):
        out[w] = words[shot, key_start + w]
    for j in range(len(ps_word)):
        out[key_words + j] = words[shot, ps_word[j]] & ps_mask[j]
    for w in range(key_words + len(ps_word), len(out)):
        out[w] = 0


@jit
def _hash(row):
    value = np.uint64(0x9E3779B97F4A7C15)
    for w in range(len(row)):
        value = (value ^ row[w]) * np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(31)
    return value


@jit
def _group(words, live, n, program, share):
    """Return the first lanes of the live shots: ``(symbols, row of each shot, lanes)``.

    With ``share``, shots with equal first lane symbols (``_lane_key``) share a
    lane, found in an open-addressing hash table; otherwise each has its own.
    """
    key_start, key_words = program.key_start, program.key_words
    ps_word, ps_mask = program.ps_word, program.ps_mask
    symbols = np.zeros((max(16, n), program.lane_words), dtype=np.uint64)
    row = np.empty(n, dtype=np.int64)
    if not share:
        for i in range(n):
            _lane_key(
                words, live[i], key_start, key_words, ps_word, ps_mask, symbols[i]
            )
            row[i] = i
        return symbols, row, n
    size = 16
    while size < 2 * n:
        size <<= 1
    table = np.full(size, -1, dtype=np.int64)
    candidate = np.zeros(program.lane_words, dtype=np.uint64)
    count = 0
    for i in range(n):
        _lane_key(words, live[i], key_start, key_words, ps_word, ps_mask, candidate)
        slot = _hash(candidate) & np.uint64(size - 1)
        while True:
            lane = table[slot]
            if lane < 0:
                table[slot] = count
                symbols[count] = candidate
                lane = count
                count += 1
                break
            same = True
            for w in range(program.lane_words):
                same &= symbols[lane, w] == candidate[w]
            if same:
                break
            slot = (slot + np.uint64(1)) & np.uint64(size - 1)
        row[i] = lane
    return symbols, row, count


@jit
def _prune(
    amps,
    symbols,
    size,
    span,
    tests,
    prune_row,
    graves,
    grave_at,
    dead,
    code_row,
    action,
):
    """Bury the lanes in which a tested output reads 1.

    The other lanes move down in order, and ``code_row`` maps codes to their new
    rows, or to the graves. Returns ``(size, graves, grave_at, dead)``.
    """
    new_row = np.empty(size, dtype=np.int64)
    kept = 0
    for r in range(size):
        dies = False
        for o in range(len(tests)):
            if tests[o] and _row_parity(symbols, r, prune_row[o]):
                dies = True
                break
        if dies:
            if dead == len(grave_at):
                graves = _grown(graves, 2 * dead, dead)
                at = np.zeros(2 * dead, dtype=np.int64)
                at[:dead] = grave_at
                grave_at = at
            graves[dead] = symbols[r]
            grave_at[dead] = action
            new_row[r] = -1 - dead
            dead += 1
            continue
        if kept != r:
            _copy_lane(amps, kept, r, span)
            symbols[kept] = symbols[r]
        new_row[r] = kept
        kept += 1
    if kept < size:
        for code in range(len(code_row)):
            if code_row[code] >= 0:
                code_row[code] = new_row[code_row[code]]
    return kept, graves, grave_at, dead


@kernel
def run_lanes(
    words, alive, first_shot, seed, share, program, poison, parallel_work, threads
):
    """Run one component on the alive shots of a chunk and add its outputs.

    ``words`` holds the chunk's shot words (row ``s`` is shot ``first_shot + s``)
    and ``program`` is the component's ``LaneProgram``. The shots' output bits get
    the component's branch parts and coins, and those that a pruning hides are
    cleared. A pruned shot leaves ``alive``, so later components skip it. The
    amplitude loops run in up to ``4 * threads`` parallel chunks once
    their work reaches ``parallel_work``.

    Returns:
        ``(deviation, groups, peak, size, dead)``: the largest normalization
        deviation, the first lanes, the most lanes at once, the live lanes at the
        end and the pruned ones.

    """
    shots = words.shape[0]
    num_actions = len(program.kind)
    out_words = program.out_words
    num_coins = program.coin_pattern.shape[0]
    # NOTE: a shot's draw slots are one per action, then its coins, 64 per slot.
    slots = num_actions + (num_coins + 63) // 64

    live = np.empty(shots, dtype=np.int64)
    n = 0
    for s in range(shots):
        if alive[s]:
            live[n] = s
            n += 1
    if n == 0:
        return 0.0, 0, 0, 0, 0
    symbols, row, size = _group(words, live, n, program, share)
    groups, peak = size, size
    amps = np.zeros((max(1, size), 1 << program.width), dtype=np.complex128)
    for r in range(size):
        amps[r, 0] = 1
    # NOTE: each live shot holds a code, and ``code_row`` maps codes to rows, or to
    # graves as ``-1 - grave``: splits and prunes rewrite the map in a step per
    # lane, and the next pass over the shots drops those in graves.
    code_row = np.arange(max(size, 1), dtype=np.int64)
    graves = np.zeros((16, program.lane_words), dtype=np.uint64)
    grave_at = np.zeros(16, dtype=np.int64)
    dead = 0
    dead_shot = np.empty(n, dtype=np.int64)
    dead_grave = np.empty(n, dtype=np.int64)
    num_dead_shots = 0
    region = 0
    deviation = 0.0
    if size and program.prune_first.any():
        size, graves, grave_at, dead = _prune(
            amps,
            symbols,
            size,
            1,
            program.prune_first,
            program.prune_row,
            graves,
            grave_at,
            dead,
            code_row,
            -1,
        )

    for a in range(num_actions):
        if size == 0:
            break
        if poison:
            _poison(amps, size, 1 << region)
        shift = program.extent[a]
        span = 1 << shift
        if shift > region:
            _clear(amps, size, 1 << region, span)
        region = shift
        kind = program.kind[a]
        x, z = np.int64(program.x_mask[a]), np.int64(program.z_mask[a])
        if kind in (ROTATE_DIAG, ROTATE):
            flip = np.empty(size, dtype=np.int64)
            for r in range(size):
                flip[r] = _row_parity(symbols, r, program.sign_row[a]) ^ (
                    program.sign_const[a]
                )
            diagonal = kind == ROTATE_DIAG
            cos, factor, coupling = (
                program.cos[a],
                program.factor[a],
                program.coupling[a],
            )
            units = size * span if diagonal else size * (span >> 1)
            chunks = _chunks(size * span, units, parallel_work, threads)
            if chunks == 1:
                _rotate_range(
                    amps, 0, units, span, flip, diagonal, x, z, cos, factor, coupling
                )
            else:
                for c in numba.prange(chunks):
                    lo, hi = units * c // chunks, units * (c + 1) // chunks
                    _rotate_range(
                        amps, lo, hi, span, flip, diagonal, x, z, cos, factor, coupling
                    )
            continue
        if not program.splits[a]:
            continue
        dormant = kind == MEASURE_DORMANT
        half = 0.5 if dormant else 0.0
        w0 = np.full(size, half)
        w1 = np.full(size, half)
        i_power = program.i_power[a]
        if not dormant:
            chunks = _chunks(size * span, size, parallel_work, threads)
            if chunks == 1:
                _weights_range(amps, 0, size, span, x, z, i_power, w0, w1)
            else:
                for c in numba.prange(chunks):
                    lo, hi = size * c // chunks, size * (c + 1) // chunks
                    _weights_range(amps, lo, hi, span, x, z, i_power, w0, w1)
        # NOTE: each lane's outcome if certain (0 or 1), else -1 and its threshold.
        certain = np.empty(size, dtype=np.int64)
        threshold = np.zeros(size, dtype=np.uint64)
        any_random = False
        for r in range(size):
            total = w0[r] + w1[r]
            if not dormant:
                d = abs(total - 1)
                # NOTE: written so that a NaN deviation sticks.
                if d > deviation or d != d:
                    deviation = d
            certain[r], threshold[r] = _cut(w0[r], w1[r])
            if certain[r] < 0:
                any_random = True
        outcome = np.empty(2 * size, dtype=np.bool_)
        new = size
        if any_random:
            seen = np.zeros(2 * size, dtype=np.bool_)
            m = 0
            for i in range(n):
                shot, r = live[i], code_row[row[i]]
                if r < 0:
                    dead_shot[num_dead_shots] = shot
                    dead_grave[num_dead_shots] = -1 - r
                    num_dead_shots += 1
                    continue
                c = certain[r]
                if c < 0:
                    u = _draw(seed, first_shot + shot, slots, a)
                    c = np.int64(u < threshold[r])
                live[m], row[m] = shot, 2 * r + c
                seen[2 * r + c] = True
                m += 1
            n = m
            for r in range(size):
                if seen[2 * r] and seen[2 * r + 1]:
                    new += 1
            if new > amps.shape[0]:
                # NOTE: a lane holds a live shot, so there are at most ``n``.
                amps = _grown(amps, min(max(new, 2 * amps.shape[0]), n), size)
            code_row = np.empty(2 * size, dtype=np.int64)
            weight = np.empty(new)
            child = size
            for r in range(size):
                code_row[2 * r] = code_row[2 * r + 1] = r
                if seen[2 * r] and seen[2 * r + 1]:
                    _copy_lane(amps, child, r, span)
                    symbols[child] = symbols[r]
                    code_row[2 * r + 1] = child
                    outcome[child], weight[child] = True, w1[r]
                    child += 1
                one = seen[2 * r + 1] and not seen[2 * r]
                outcome[r], weight[r] = one, w1[r] if one else w0[r]
        else:
            weight = np.empty(size)
            for r in range(size):
                outcome[r] = certain[r] == 1
                weight[r] = w1[r] if outcome[r] else w0[r]
        size = new
        peak = max(peak, size)
        if not dormant:
            pivot = program.pivot[a]
            chunks = _chunks(size * span, size, parallel_work, threads)
            if chunks == 1:
                _project_range(
                    amps, 0, size, span, x, z, pivot, i_power, outcome, weight
                )
            else:
                for c in numba.prange(chunks):
                    lo, hi = size * c // chunks, size * (c + 1) // chunks
                    _project_range(
                        amps, lo, hi, span, x, z, pivot, i_power, outcome, weight
                    )
        bit = program.branch[a]
        if bit >= 0:
            for r in range(size):
                if outcome[r]:
                    symbols[r, bit >> 6] |= np.uint64(1) << np.uint64(bit & 63)
        if program.pruned[a]:
            size, graves, grave_at, dead = _prune(
                amps,
                symbols,
                size,
                1 << region,
                program.prune_outputs[a],
                program.prune_row,
                graves,
                grave_at,
                dead,
                code_row,
                a,
            )

    # NOTE: each entry (a live lane or a grave) has a branch pattern on the outputs
    # and hides the outputs that its pruning makes unknown; every shot takes its
    # entry's.
    entries = size + dead
    pattern = np.zeros((entries, len(out_words)), dtype=np.uint64)
    hidden = np.zeros((entries, len(out_words)), dtype=np.uint64)
    for e in range(entries):
        lane_symbols = symbols if e < size else graves
        r = e if e < size else e - size
        for o in range(len(program.output_bit)):
            w = program.output_word[o]
            one = np.uint64(1) << np.uint64(program.output_bit[o])
            if _row_parity(lane_symbols, r, program.branch_row[o]):
                pattern[e, w] ^= one
            if e >= size and program.ready[o] > grave_at[r]:
                hidden[e, w] |= one
    for i in range(n + num_dead_shots):
        if i < n:
            shot, e = live[i], code_row[row[i]]
            if e < 0:
                e = size - 1 - e
        else:
            shot, e = dead_shot[i - n], size + dead_grave[i - n]
        if e >= size:
            alive[shot] = False
        for w in range(len(out_words)):
            words[shot, out_words[w]] ^= pattern[e, w]
        coins = np.uint64(0)
        for j in range(num_coins):
            if j % 64 == 0:
                coins = _draw(seed, first_shot + shot, slots, num_actions + j // 64)
            if (coins >> np.uint64(j % 64)) & np.uint64(1):
                for w in range(len(out_words)):
                    words[shot, out_words[w]] ^= program.coin_pattern[j, w]
        for w in range(len(out_words)):
            words[shot, out_words[w]] &= ~hidden[e, w]
    return deviation, groups, peak, size, dead


# NOTE: tests set this to fill every amplitude the executor must not read with NaN.
_POISON = False
