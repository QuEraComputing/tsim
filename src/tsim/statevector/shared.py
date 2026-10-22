"""Trajectory-sharing executor for statevector components, on the CPU with numba.

Shots that agree on every symbol that reaches the amplitudes share one lane. The
only per-shot input to the amplitudes is the sign of a rotation (outputs are
parities of the final symbols), so a shot's grouping key is the f-bit part of
every rotation sign. With postselection the key also holds the f-bit part of
every postselected output, so that a lane is pruned as a whole as soon as one of
its postselected outputs is known to be 1: at the action that sets the output's
last branch bit, or before the first action for an output of f-bits only.

A lane holds amplitudes, the symbols of one of its shots (its f-bits agree with
every member on the key, and its branch bits are the lane's outcomes), a shot
count and a range in the key-sorted order of the shots. At a measurement the lane
draws how many of its shots see the rarer outcome (a binomial whose chance of a
first hit is resolved to 2^-64) and splits: the child takes the last n shots of
the range. Shots in a lane agree on the key, so they are exchangeable, and the
sort breaks ties at random, so each range is in uniformly random order: this
assignment gives every shot independent outputs. Pruned lanes keep their range,
outcomes and pruning action but drop their amplitudes. A dormant measurement
splits lanes the same way (p = 1/2) when its branch reaches a key sign; otherwise
it is a fair coin drawn per shot at the end.

One numba function samples a batch without the GIL: it groups the shots, runs
the tape on the live lanes and the ``2^extent`` amplitudes each action can
touch, and evaluates the outputs. When the extent grows, the newly touched
amplitudes are cleared, because a slot reused for another lane can hold stale
values there. After the tape each shot's outputs are evaluated from its own
f-bits, its lane's outcomes and its own dormant coins, XOR its reference bit; a
pruned shot's outputs that need a branch bit set after its pruning action are
False.
"""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np

import tsim.utils.kernels as kernels
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
from tsim.utils.rng import GOLDEN, next_skip, splitmix64

_TINY = 1e-30
_I_POWERS = (1, 1j, -1, -1j)


class SharingLayout(NamedTuple):
    """Which symbols the shots of a lane must share, and which outputs prune them.

    Attributes:
        key_rows: f-bit masks whose parities form the grouping key, a basis of the
            f-parts of every rotation sign and every postselected output,
            shape ``(K, num_words)``.
        lane_dormant: Dormant measurements that split lanes, shape ``(A,)``.
        prune_first: The postselected outputs of f-bits only, tested before the
            first action, shape ``(num_outputs,)``.
        pruned: Actions after which some postselected output is tested,
            shape ``(A,)``.
        prune_outputs: The postselected outputs tested after each action,
            shape ``(A, num_outputs)``.
        output_const: Constant of each output XOR its reference bit, shape
            ``(num_outputs,)``.
        lane_branches: Branch bits a lane carries, shape ``(num_words,)``.
        shot_dormant: Branch bits of the dormant measurements drawn per shot,
            shape ``(num_words,)``.
        ready: The action that sets each output's last branch bit, -1 for an
            output of f-bits only, shape ``(num_outputs,)``.
        i_power: ``i^|x∧z|`` of each action's Pauli ``X^x Z^z``, shape ``(A,)``.
        factor: Each action's diagonal rotation factors ``cos θ ∓ i sin θ``,
            shape ``(A, 2)``.
        coupling: Each action's rotation coupling ``-i sin θ i^|x∧z|``, shape
            ``(A,)``.

    """

    key_rows: np.ndarray
    lane_dormant: np.ndarray
    prune_first: np.ndarray
    pruned: np.ndarray
    prune_outputs: np.ndarray
    output_const: np.ndarray
    lane_branches: np.ndarray
    shot_dormant: np.ndarray
    ready: np.ndarray
    i_power: np.ndarray
    factor: np.ndarray
    coupling: np.ndarray


def _unpack(words: np.ndarray, num_words: int) -> np.ndarray:
    shifts = np.arange(32, dtype=np.uint32)
    bits = (words[..., None] >> shifts) & 1
    return bits.reshape(*words.shape[:-1], 32 * num_words).astype(bool)


def _pack(bits: np.ndarray, num_words: int) -> np.ndarray:
    padded = np.zeros((*bits.shape[:-1], 32 * num_words), dtype=np.uint64)
    padded[..., : bits.shape[-1]] = bits
    shifts = np.arange(32, dtype=np.uint64)
    words = padded.reshape(*bits.shape[:-1], num_words, 32) << shifts
    return words.sum(axis=-1).astype(np.uint32)


def sharing_layout(
    component: StatevectorComponent,
    postselected: np.ndarray | None = None,
    flipped: np.ndarray | None = None,
) -> SharingLayout:
    """Return the sharing layout of a component.

    Args:
        component: The compiled component.
        postselected: Whether each output slot is postselected, shape
            ``(num_outputs,)``. ``None`` prunes nothing.
        flipped: Reference bit of each output slot, XOR-ed into the output before
            the postselection test and before a pruned shot's later outputs are
            cleared, shape ``(num_outputs,)``. ``None`` flips nothing.

    """
    tape = component.tape
    num_words = component.num_words
    num_f = len(tape.f_selection)
    kind, branch = tape.kind, tape.branch
    num_outputs = component.num_outputs
    sign_bits = _unpack(tape.sign_row, num_words)
    output_bits = _unpack(tape.output_row, num_words)
    no_outputs = np.zeros(num_outputs, dtype=bool)
    postselected = np.asarray(
        no_outputs if postselected is None else postselected, dtype=bool
    )
    flipped = np.asarray(no_outputs if flipped is None else flipped, dtype=bool)

    setter = np.full(32 * num_words, -1, dtype=np.int32)
    setter[branch[branch >= 0]] = np.flatnonzero(branch >= 0)
    ready = np.max(np.where(output_bits, setter[None, :], -1), axis=1, initial=-1)
    prune_outputs = np.zeros((len(kind), num_outputs), dtype=bool)
    for o in np.flatnonzero(postselected & (ready >= 0)):
        prune_outputs[ready[o], o] = True

    rotation = np.isin(kind, (ROTATE_DIAG, ROTATE))
    keyed_bits = np.vstack([sign_bits[rotation], output_bits[postselected]])
    key_bits = keyed_bits[:, :num_f]
    key_bits = key_bits[key_bits.any(axis=1)]
    if len(key_bits):
        key_bits, _ = find_basis(key_bits.astype(np.uint8))
        key_bits = np.asarray(key_bits, dtype=bool)

    # NOTE: a pruning action sets a branch bit of a postselected output, so its
    # branch reaches the key.
    reaches_key = np.asarray(np.any(keyed_bits, axis=0), dtype=bool)
    dormant = kind == MEASURE_DORMANT
    lane_dormant = dormant & reaches_key[branch]
    active = np.isin(kind, (MEASURE_DIAG, MEASURE))
    lane_bits = np.zeros(32 * num_words, dtype=bool)
    lane_bits[branch[active | lane_dormant]] = True
    shot_bits = np.zeros(32 * num_words, dtype=bool)
    shot_bits[branch[dormant & ~lane_dormant]] = True

    quarter_turns = [
        int(x & z).bit_count() & 3
        for x, z in zip(tape.x_mask, tape.z_mask, strict=True)
    ]
    i_power = np.array([_I_POWERS[q] for q in quarter_turns], dtype=np.complex128)
    cos, sin = tape.cos, tape.sin

    return SharingLayout(
        key_rows=_pack(key_bits, num_words).reshape(-1, num_words),
        lane_dormant=lane_dormant,
        prune_first=postselected & (ready < 0),
        pruned=np.asarray(prune_outputs.any(axis=1)),
        prune_outputs=prune_outputs,
        output_const=tape.output_const ^ flipped,
        lane_branches=_pack(lane_bits, num_words),
        shot_dormant=_pack(shot_bits, num_words),
        ready=ready.astype(np.int64),
        i_power=i_power,
        factor=np.stack([cos - 1j * sin, cos + 1j * sin], axis=1),
        coupling=-1j * sin * i_power,
    )


@jit
def _parity(value):
    """Return the parity of the bits of a non-negative integer below 2^32."""
    value ^= value >> 16
    value ^= value >> 8
    value ^= value >> 4
    value ^= value >> 2
    value ^= value >> 1
    return value & 1


@jit
def _chunks(work, parallel_work, threads):
    """Return how many chunks a loop of ``work`` runs in: 1 below ``parallel_work``."""
    if work < parallel_work:
        return 1
    return min(work, 4 * threads)


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
def _grown(amps, rows, lanes, span):
    """Return ``amps`` with ``rows`` rows, keeping the lanes' first ``span``."""
    out = np.zeros((rows, amps.shape[1]), dtype=amps.dtype)
    for lane in range(lanes):
        for i in range(span):
            out[lane, i] = amps[lane, i]
    return out


@jit
def _pack_range(f, selection, out, lo, hi):
    """Pack the selected f-bits of shots ``lo … hi-1``: bit j is in word j // 32."""
    for s in range(lo, hi):
        for j in range(len(selection)):
            if f[s, selection[j]]:
                out[s, j >> 5] |= np.uint32(1) << np.uint32(j & 31)


@jit
def _key_range(symbols, key_rows, out, lo, hi):
    """Pack the parities of shots ``lo … hi-1`` with the key rows into words."""
    for s in range(lo, hi):
        for k in range(key_rows.shape[0]):
            acc = 0
            for w in range(key_rows.shape[1]):
                acc ^= symbols[s, w] & key_rows[k, w]
            if _parity(acc):
                out[s, k >> 5] |= np.uint32(1) << np.uint32(k & 31)


@jit
def _signs(symbols, lanes, row, const):
    """Return each lane's sign bit: ``const ^ parity(symbols[l] & row)``."""
    flip = np.zeros(lanes, dtype=np.int64)
    for lane in range(lanes):
        acc = 0
        for w in range(len(row)):
            acc ^= symbols[lane, w] & row[w]
        flip[lane] = _parity(acc) ^ const
    return flip


@jit
def _rotate_range(amps, lo, hi, shift, flip, diagonal, x, z, cos, factor, coupling):
    """Apply one rotation to the entries ``lo … hi-1`` of ``lanes × 2^shift``.

    A diagonal rotation multiplies amplitude i by ``factor[sign ^ parity(i & z)]``,
    the others mix amplitudes i and ``i ^ x`` with ``cos`` and ``±coupling``.
    """
    span = 1 << shift
    low = x & -x
    for t in range(lo, hi):
        lane, i = t >> shift, t & (span - 1)
        if diagonal:
            amps[lane, i] *= factor[flip[lane] ^ _parity(i & z)]
            continue
        if i & low:
            continue
        j = i ^ x
        c = -coupling if flip[lane] else coupling
        p, q = amps[lane, i], amps[lane, j]
        amps[lane, i] = cos * p + c * (-q if _parity(j & z) else q)
        amps[lane, j] = cos * q + c * (-p if _parity(i & z) else p)


@jit
def _weights_range(amps, lo, hi, span, x, z, i_power, w0, w1):
    """Write the outcome weights of measuring ``X^x Z^z`` for lanes ``lo … hi-1``."""
    for lane in range(lo, hi):
        for i in range(span):
            a = amps[lane, i]
            if x == 0:
                if _parity(i & z):
                    w1[lane] += a.real**2 + a.imag**2
                else:
                    w0[lane] += a.real**2 + a.imag**2
            else:
                j = i ^ x
                b = i_power * amps[lane, j]
                if _parity(j & z):
                    b = -b
                plus, minus = a + b, a - b
                w0[lane] += (plus.real**2 + plus.imag**2) / 4
                w1[lane] += (minus.real**2 + minus.imag**2) / 4


@jit
def _project_range(amps, lo, hi, span, x, z, pivot, i_power, outcome, weight):
    """Project lanes ``lo … hi-1`` onto their outcome and free the pivot.

    The pivot-1 half of the amplitudes becomes 0: the planner moved the measured
    Pauli onto the pivot, so the pivot ends in ``|0⟩``.
    """
    bit = 1 << pivot
    for lane in range(lo, hi):
        if x == 0:
            scale = 1 / np.sqrt(max(weight[lane], _TINY))
        else:
            scale = 1 / np.sqrt(max(2 * weight[lane], _TINY))
        for i in range(span):
            if i & bit:
                continue
            if x == 0:
                j = i ^ bit
                keep = _parity(i & z) == outcome[lane]
                amps[lane, i] = (amps[lane, i] if keep else amps[lane, j]) * scale
            else:
                j = i ^ x
                b = i_power * amps[lane, j]
                if _parity(j & z):
                    b = -b
                if outcome[lane]:
                    b = -b
                amps[lane, i] = (amps[lane, i] + b) * scale
            amps[lane, j] = 0


@jit
def _draw_ones(count, w0, w1, state):
    """Return how many of each lane's shots see outcome 1.

    The rarer outcome's count is a Binomial(count, q) drawn by geometric skips,
    so the chance of a first hit, 1 - (1 - q)^count, is exact to about 2^-64.
    """
    ones = np.zeros(len(count), dtype=np.int64)
    for lane in range(len(count)):
        total = w0[lane] + w1[lane]
        q = min(w0[lane], w1[lane]) / total if total > _TINY else 0.0
        hits = 0
        if q > 0:
            log_keep = np.log1p(-q)
            position = -1.0
            while True:
                state, skip = next_skip(state, log_keep)
                position += 1.0 + skip
                if position >= count[lane]:
                    break
                hits += 1
        ones[lane] = hits if w1[lane] <= w0[lane] else count[lane] - hits
    return ones


@jit
def _group(keys, state):
    """Sort shots by key; return ``(order, start, count, state)`` of the groups.

    Ties are broken at random (a random permutation, then stable sorts), so the
    shots of a group appear in uniformly random order.
    """
    num_shots = keys.shape[0]
    order = np.arange(num_shots)
    for i in range(num_shots - 1, 0, -1):
        state, z = splitmix64(state)
        j = z % np.uint64(i + 1)
        order[i], order[j] = order[j], order[i]
    for column in range(keys.shape[1]):
        order = order[np.argsort(keys[order, column], kind="mergesort")]
    start = np.empty(num_shots, dtype=np.int64)
    groups = 0
    for p in range(num_shots):
        new = p == 0
        for column in range(keys.shape[1]):
            new |= keys[order[p], column] != keys[order[p - 1], column]
        if new:
            start[groups] = p
            groups += 1
    start = start[:groups]
    count = np.empty(groups, dtype=np.int64)
    for g in range(groups):
        count[g] = (start[g + 1] if g + 1 < groups else num_shots) - start[g]
    return order, start, count, state


@jit
def _prune(amps, span, size, dead, tests, output_row, output_const, lanes, graves, a):
    """Move the lanes with a tested output of 1 to the dead lanes; return the counts.

    ``tests`` marks the outputs to test, ``lanes`` holds the live lanes' symbols,
    starts and counts, and ``graves`` the same and the pruning action of the dead
    lanes. The live lanes keep their order in rows ``0 … kept-1``.
    """
    symbols, start, count = lanes
    dead_symbols, dead_start, dead_count, dead_action = graves
    kept = 0
    for lane in range(size):
        dies = False
        for o in range(len(tests)):
            if not tests[o]:
                continue
            acc = 0
            for w in range(symbols.shape[1]):
                acc ^= symbols[lane, w] & output_row[o, w]
            if _parity(acc) ^ output_const[o]:
                dies = True
                break
        if dies:
            dead_symbols[dead] = symbols[lane]
            dead_start[dead], dead_count[dead] = start[lane], count[lane]
            dead_action[dead] = a
            dead += 1
            continue
        if kept != lane:
            _copy_lane(amps, kept, lane, span)
            symbols[kept] = symbols[lane]
            start[kept], count[kept] = start[lane], count[lane]
        kept += 1
    return kept, dead


@jit
def _poison(amps, size, span):
    """Fill every amplitude the executor must not read with NaN (for tests)."""
    amps[:size, span:] = np.nan
    amps[size:] = np.nan


@jit
def _shot_range(
    symbols, order, owner, entry_symbols, lane_branches, dormant, seed, out, lo, hi
):
    """Write the symbols of the shots at positions ``lo … hi-1``.

    A shot's symbols are its f-bits, its lane's branches and its dormant coins.
    """
    num_words = symbols.shape[1]
    for position in range(lo, hi):
        shot, entry = order[position], owner[position]
        for w in range(num_words):
            word = symbols[shot, w] | (entry_symbols[entry, w] & lane_branches[w])
            if dormant[w]:
                n = np.uint64(position * num_words + w)
                _, coin = splitmix64(seed + n * GOLDEN)
                word |= np.uint32(coin & np.uint64(0xFFFFFFFF)) & dormant[w]
            out[shot, w] = word


@jit
def _output_range(
    shot_symbols, output_row, output_const, ready, pruned_at, out, lo, hi
):
    """Write the outputs of shots ``lo … hi-1``; outputs set after pruning are 0."""
    for s in range(lo, hi):
        for o in range(output_row.shape[0]):
            acc = 0
            for w in range(output_row.shape[1]):
                acc ^= shot_symbols[s, w] & output_row[o, w]
            out[s, o] = bool(_parity(acc) ^ output_const[o]) and (
                ready[o] <= pruned_at[s]
            )


@kernel
def _execute(f, tape, layout, width, seed, share, parallel_work, threads, poison):
    """Sample one batch of shots with shared trajectories.

    ``tape`` is a ``Tape`` and ``layout`` a ``SharingLayout``. Lanes are rows
    ``0 … size-1`` of arrays with a row per shot (every lane holds a shot), but
    the amplitudes grow as lanes split. Each loop runs in chunks across threads
    once its work reaches ``parallel_work``.

    Returns:
        ``(samples, deviation, groups, peak, lanes, dead)``.

    """
    # NOTE: numba's parallel loops cannot read the fields of a named tuple.
    selection, output_row = tape.f_selection, tape.output_row
    output_const, ready = layout.output_const, layout.ready
    key_rows, lane_branches, shot_dormant = (
        layout.key_rows,
        layout.lane_branches,
        layout.shot_dormant,
    )
    shots, num_words = f.shape[0], lane_branches.shape[0]
    num_actions = tape.kind.shape[0]
    # NOTE: SplitMix64 counts up from its state, so seeds that differ by a few
    # GOLDEN steps would share outputs; their mixed output starts a fresh stream.
    _, state = splitmix64(np.uint64(seed))

    symbols = np.zeros((shots, num_words), dtype=np.uint32)
    num_key_words = (key_rows.shape[0] + 31) >> 5 if share else 0
    keys = np.zeros((shots, num_key_words), dtype=np.uint32)
    key_work = key_rows.size if share else 0
    chunks = _chunks(shots * (len(selection) + key_work), parallel_work, threads)
    if chunks == 1:
        _pack_range(f, selection, symbols, 0, shots)
        if share:
            _key_range(symbols, key_rows, keys, 0, shots)
    else:
        for c in numba.prange(chunks):
            lo, hi = shots * c // chunks, shots * (c + 1) // chunks
            _pack_range(f, selection, symbols, lo, hi)
            if share:
                _key_range(symbols, key_rows, keys, lo, hi)
    if share:
        order, start, count, state = _group(keys, state)
    else:
        order = np.arange(shots)
        start = np.arange(shots)
        count = np.ones(shots, dtype=np.int64)
    groups = len(start)

    # NOTE: lane and dead-lane rows are written before they are read.
    lane_symbols = np.empty((shots, num_words), dtype=np.uint32)
    lane_start = np.empty(shots, dtype=np.int64)
    lane_count = np.empty(shots, dtype=np.int64)
    amps = np.zeros((min(max(16, 2 * groups), shots), 1 << width), dtype=np.complex128)
    for g in range(groups):
        lane_symbols[g] = symbols[order[start[g]]]
        lane_start[g], lane_count[g] = start[g], count[g]
        amps[g, 0] = 1
    lanes = (lane_symbols, lane_start, lane_count)
    graves = (
        np.empty((shots, num_words), dtype=np.uint32),
        np.empty(shots, dtype=np.int64),
        np.empty(shots, dtype=np.int64),
        np.empty(shots, dtype=np.int64),
    )
    size, dead = _prune(
        amps,
        1,
        groups,
        0,
        layout.prune_first,
        output_row,
        output_const,
        lanes,
        graves,
        -1,
    )
    peak, region, deviation = groups, 0, 0.0

    for a in range(num_actions):
        if size == 0:
            break
        if poison:
            _poison(amps, size, 1 << region)
        shift = tape.extent[a]
        span = 1 << shift
        if shift > region:
            _clear(amps, size, 1 << region, span)
        region = shift
        kind = tape.kind[a]
        x, z = np.int64(tape.x_mask[a]), np.int64(tape.z_mask[a])
        if kind in (ROTATE_DIAG, ROTATE):
            flip = _signs(
                lane_symbols, size, tape.sign_row[a], np.int64(tape.sign_const[a])
            )
            diagonal = kind == ROTATE_DIAG
            cos, factor, coupling = tape.cos[a], layout.factor[a], layout.coupling[a]
            work = size * span
            chunks = _chunks(work, parallel_work, threads)
            if chunks == 1:
                _rotate_range(
                    amps, 0, work, shift, flip, diagonal, x, z, cos, factor, coupling
                )
            else:
                for c in numba.prange(chunks):
                    lo, hi = work * c // chunks, work * (c + 1) // chunks
                    _rotate_range(
                        amps, lo, hi, shift, flip, diagonal, x, z, cos, factor, coupling
                    )
            continue
        dormant = kind == MEASURE_DORMANT
        if dormant and not layout.lane_dormant[a]:
            continue
        half = 0.5 if dormant else 0.0
        w0 = np.full(size, half)
        w1 = np.full(size, half)
        i_power, pivot = layout.i_power[a], tape.pivot[a]
        if not dormant:
            chunks = _chunks(size * span, parallel_work, threads)
            if chunks == 1:
                _weights_range(amps, 0, size, span, x, z, i_power, w0, w1)
            else:
                for c in numba.prange(chunks):
                    lo, hi = size * c // chunks, size * (c + 1) // chunks
                    _weights_range(amps, lo, hi, span, x, z, i_power, w0, w1)
            for lane in range(size):
                d = abs(w0[lane] + w1[lane] - 1)
                # NOTE: written so that a NaN deviation sticks.
                if d > deviation or d != d:
                    deviation = d
        state, draw = splitmix64(state)
        ones = _draw_ones(lane_count[:size], w0, w1, draw)

        new = size
        for lane in range(size):
            if 0 < ones[lane] < lane_count[lane]:
                new += 1
        if new > amps.shape[0]:
            rows = min(max(new, 2 * amps.shape[0]), shots)
            amps = _grown(amps, rows, size, span)
        outcome = np.zeros(new, dtype=np.bool_)
        weight = np.zeros(new)
        child = size
        for lane in range(size):
            n, total = ones[lane], lane_count[lane]
            outcome[lane] = n == total
            weight[lane] = w1[lane] if n == total else w0[lane]
            if 0 < n < total:
                _copy_lane(amps, child, lane, span)
                lane_symbols[child] = lane_symbols[lane]
                lane_count[child] = n
                lane_start[child] = lane_start[lane] + total - n
                lane_count[lane] = total - n
                outcome[child] = True
                weight[child] = w1[lane]
                child += 1
        size = new
        if not dormant:
            chunks = _chunks(size * span, parallel_work, threads)
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
        bit = tape.branch[a]
        if bit >= 0:
            for lane in range(size):
                if outcome[lane]:
                    lane_symbols[lane, bit >> 5] |= np.uint32(1) << np.uint32(bit & 31)
        peak = max(peak, size)
        if layout.pruned[a]:
            tests = layout.prune_outputs[a]
            size, dead = _prune(
                amps,
                span,
                size,
                dead,
                tests,
                output_row,
                output_const,
                lanes,
                graves,
                a,
            )

    # NOTE: the live and dead lanes' ranges tile the sorted positions.
    dead_symbols, dead_start, dead_count, dead_action = graves
    entry_symbols = np.zeros((size + dead, num_words), dtype=np.uint32)
    entry_pruned = np.empty(size + dead, dtype=np.int64)
    owner = np.empty(shots, dtype=np.int64)
    for e in range(size + dead):
        if e < size:
            entry_symbols[e] = lane_symbols[e]
            entry_pruned[e] = num_actions
            first, last = lane_start[e], lane_start[e] + lane_count[e]
        else:
            entry_symbols[e] = dead_symbols[e - size]
            entry_pruned[e] = dead_action[e - size]
            first = dead_start[e - size]
            last = first + dead_count[e - size]
        for p in range(first, last):
            owner[p] = e
    pruned_at = np.empty(shots, dtype=np.int64)
    for p in range(shots):
        pruned_at[order[p]] = entry_pruned[owner[p]]
    state, coins = splitmix64(state)
    shot_symbols = np.empty_like(symbols)
    samples = np.zeros((shots, output_row.shape[0]), dtype=np.bool_)
    chunks = _chunks(shots * (num_words + output_row.size), parallel_work, threads)
    if chunks == 1:
        _shot_range(
            symbols,
            order,
            owner,
            entry_symbols,
            lane_branches,
            shot_dormant,
            coins,
            shot_symbols,
            0,
            shots,
        )
        _output_range(
            shot_symbols, output_row, output_const, ready, pruned_at, samples, 0, shots
        )
    else:
        for c in numba.prange(chunks):
            lo, hi = shots * c // chunks, shots * (c + 1) // chunks
            _shot_range(
                symbols,
                order,
                owner,
                entry_symbols,
                lane_branches,
                shot_dormant,
                coins,
                shot_symbols,
                lo,
                hi,
            )
        for c in numba.prange(chunks):
            lo, hi = shots * c // chunks, shots * (c + 1) // chunks
            _output_range(
                shot_symbols,
                output_row,
                output_const,
                ready,
                pruned_at,
                samples,
                lo,
                hi,
            )
    return samples, deviation, groups, peak, size, dead


# NOTE: tests set this to fill every amplitude the executor must not read with NaN.
_POISON = False


def lane_bytes(component: StatevectorComponent) -> int:
    """Return the peak memory of one lane of ``sample_shared_component``.

    That is three rows of amplitudes (the old and the new array while lanes
    grow) and the per-shot symbols, ranges and outputs.
    """
    amplitudes = 3 * 2**component.width * np.dtype(np.complex128).itemsize
    return amplitudes + 24 * component.num_words + 128 + component.num_outputs


def lane_bound(layout: SharingLayout) -> int:
    """Return the most lanes a batch with sharing can need, however many shots.

    The shots of a lane agree on the key, of ``len(key_rows)`` bits, and every
    measurement that splits lanes, one per lane branch bit, at most doubles them.
    """
    splits = sum(int(word).bit_count() for word in layout.lane_branches)
    return 2 ** (len(layout.key_rows) + splits)


def sample_shared_component(
    component: StatevectorComponent,
    layout: SharingLayout,
    f_params: np.ndarray,
    seed: int,
    share: bool = True,
) -> tuple[np.ndarray, float, dict]:
    """Sample one component's outputs, with shared trajectories if ``share``.

    Args:
        component: The compiled component.
        layout: Its sharing layout.
        f_params: Error-basis bits of every shot, shape ``(B, num_f_global)``.
        seed: Seed of the batch's random numbers, below 2^64.
        share: Whether shots with equal keys share a lane; otherwise every shot
            has its own.

    Returns:
        ``(samples, max_norm_deviation, stats)``, where ``samples`` has shape
        ``(B, num_outputs)`` and ``stats`` counts groups and lanes.

    """
    f = np.asarray(f_params)
    samples, deviation, groups, peak, lanes, dead = _execute(
        f.shape[0] << component.width,
        f,
        component.tape,
        layout,
        component.width,
        np.uint64(seed),
        share,
        kernels.PARALLEL_WORK,
        numba.get_num_threads(),
        _POISON,
    )
    stats = {"groups": groups, "peak lanes": peak, "lanes": lanes, "dead": dead}
    return samples, float(deviation), stats
