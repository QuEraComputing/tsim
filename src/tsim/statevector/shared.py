"""Trajectory-sharing executor for statevector components, on the CPU with numba.

Shots that agree on every symbol that reaches the amplitudes share one lane. The
only per-shot input to the amplitudes is the sign of a rotation (outputs are
parities of the final symbols), so a shot's grouping key is the f-bit part of
every rotation sign. With postselection the key also holds the f-bit part of
every postselected output, so that a lane is pruned as a whole as soon as one of
its postselected outputs is known to be 1: at the action that sets the output's
last branch bit (the first measurement for an output of f-bits only).

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

Lanes live in growable arrays, and a Python loop runs the tape: each action calls
a numba kernel on the live lanes and on the ``2^extent`` amplitudes the action
can touch. When the extent grows, the newly touched amplitudes are
cleared, because a slot reused for another lane can hold stale values there.
After the tape each shot's outputs are evaluated from its own f-bits, its lane's
outcomes and its own dormant coins; a pruned shot's outputs that need a branch
bit set after its pruning action are False.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import numba
import numpy as np

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


class Tape(NamedTuple):
    """A component's action tape as NumPy arrays (see ``StatevectorComponent``)."""

    kind: np.ndarray
    x_mask: np.ndarray
    z_mask: np.ndarray
    cos: np.ndarray
    sin: np.ndarray
    pivot: np.ndarray
    extent: np.ndarray
    branch: np.ndarray
    sign_const: np.ndarray
    sign_row: np.ndarray
    output_const: np.ndarray
    output_row: np.ndarray
    f_selection: np.ndarray


class SharingLayout(NamedTuple):
    """Which symbols the shots of a lane must share, and the tape they run.

    Attributes:
        tape: The component's actions and outputs.
        key_rows: f-bit masks whose parities form the grouping key, a basis of the
            f-parts of every rotation sign and every postselected output,
            shape ``(K, num_words)``.
        lane_dormant: Dormant measurements that split lanes, shape ``(A,)``.
        pruned: Actions after which some postselected output is tested,
            shape ``(A,)``.
        prune_outputs: The postselected outputs tested after each action,
            shape ``(A, num_outputs)``.
        prune_flip: Reference bit XOR-ed into each output before the
            postselection test, shape ``(num_outputs,)``.
        lane_branches: Branch bits a lane carries, shape ``(num_words,)``.
        shot_dormant: Branch bits of the dormant measurements drawn per shot,
            shape ``(num_words,)``.
        ready: The action that sets each output's last branch bit, -1 for an
            output of f-bits only, shape ``(num_outputs,)``.

    """

    tape: Tape
    key_rows: np.ndarray
    lane_dormant: np.ndarray
    pruned: np.ndarray
    prune_outputs: np.ndarray
    prune_flip: np.ndarray
    lane_branches: np.ndarray
    shot_dormant: np.ndarray
    ready: np.ndarray


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
            the postselection test (``use_detector_reference_sample``), shape
            ``(num_outputs,)``. ``None`` flips nothing.

    """
    tape = Tape(*(np.array(getattr(component, field)) for field in Tape._fields))
    num_words = component.num_words
    num_f = len(tape.f_selection)
    kind, branch = tape.kind, tape.branch
    num_actions = len(kind)
    sign_bits = _unpack(tape.sign_row, num_words)
    output_bits = _unpack(tape.output_row, num_words)
    num_outputs = component.num_outputs
    postselected = np.asarray(
        np.zeros(num_outputs) if postselected is None else postselected, dtype=bool
    )
    flipped = np.asarray(
        np.zeros(num_outputs) if flipped is None else flipped, dtype=bool
    )

    setter = np.full(32 * num_words, -1, dtype=np.int32)
    setter[branch[branch >= 0]] = np.flatnonzero(branch >= 0)
    ready = np.max(np.where(output_bits, setter[None, :], -1), axis=1, initial=-1)
    measurements = np.flatnonzero(np.isin(kind, (MEASURE_DIAG, MEASURE)))
    if len(measurements) == 0:
        measurements = np.flatnonzero(kind == MEASURE_DORMANT)
    first_measurement = measurements[0] if len(measurements) else -1
    prune_at = np.where(ready >= 0, ready, first_measurement)
    prune_outputs = np.zeros((num_actions, num_outputs), dtype=bool)
    for o in np.flatnonzero(postselected & (prune_at >= 0)):
        prune_outputs[prune_at[o], o] = True
    pruned = np.asarray(prune_outputs.any(axis=1))

    rotation = np.isin(kind, (ROTATE_DIAG, ROTATE))
    keyed_bits = np.vstack([sign_bits[rotation], output_bits[postselected]])
    key_bits = keyed_bits[:, :num_f]
    key_bits = key_bits[key_bits.any(axis=1)]
    if len(key_bits):
        key_bits, _ = find_basis(key_bits.astype(np.uint8))
        key_bits = np.asarray(key_bits, dtype=bool)

    reaches_key = np.asarray(np.any(keyed_bits, axis=0), dtype=bool)
    dormant = kind == MEASURE_DORMANT
    lane_dormant = dormant & (pruned | ((branch >= 0) & reaches_key[branch]))
    active = np.isin(kind, (MEASURE_DIAG, MEASURE))
    lane_bits = np.zeros(32 * num_words, dtype=bool)
    lane_bits[branch[(active | lane_dormant) & (branch >= 0)]] = True
    shot_bits = np.zeros(32 * num_words, dtype=bool)
    shot_bits[branch[dormant & ~lane_dormant & (branch >= 0)]] = True

    return SharingLayout(
        tape=tape,
        key_rows=_pack(key_bits, num_words).reshape(-1, num_words),
        lane_dormant=lane_dormant,
        pruned=pruned,
        prune_outputs=prune_outputs,
        prune_flip=flipped,
        lane_branches=_pack(lane_bits, num_words),
        shot_dormant=_pack(shot_bits, num_words),
        ready=ready.astype(np.int64),
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


@kernel
def _row_parities(symbols, rows):
    """Return ``parity(symbols[i] & rows[r])`` for every row i and r."""
    out = np.zeros((symbols.shape[0], rows.shape[0]), dtype=np.bool_)
    for i in numba.prange(symbols.shape[0]):
        for r in range(rows.shape[0]):
            acc = 0
            for w in range(rows.shape[1]):
                acc ^= symbols[i, w] & rows[r, w]
            out[i, r] = _parity(acc)
    return out


@kernel
def _key_words(symbols, key_rows):
    """Pack the parities of each shot's symbols with the key rows into words."""
    num_keys = key_rows.shape[0]
    out = np.zeros((symbols.shape[0], (num_keys + 31) >> 5), dtype=np.uint32)
    for s in numba.prange(symbols.shape[0]):
        for k in range(num_keys):
            acc = 0
            for w in range(key_rows.shape[1]):
                acc ^= symbols[s, w] & key_rows[k, w]
            if _parity(acc):
                out[s, k >> 5] |= np.uint32(1) << np.uint32(k & 31)
    return out


@kernel
def _pack_symbols(f, selection, num_words):
    """Pack each shot's selected f-bits: bit j is bit j % 32 of word j // 32."""
    out = np.zeros((f.shape[0], num_words), dtype=np.uint32)
    for s in numba.prange(f.shape[0]):
        for j in range(len(selection)):
            if f[s, selection[j]]:
                out[s, j >> 5] |= np.uint32(1) << np.uint32(j & 31)
    return out


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


@kernel
def _rotate_diag(amps, lanes, extent, z, symbols, sign_row, sign_const, factor):
    """Multiply amplitude i of lane l by ``factor[sign(l) ^ parity(i & z)]``."""
    flip = _signs(symbols, lanes, sign_row, sign_const)
    span = 1 << extent
    for t in numba.prange(lanes * span):
        lane, i = t >> extent, t & (span - 1)
        amps[lane, i] *= factor[flip[lane] ^ _parity(i & z)]


@kernel
def _rotate_pair(
    amps, lanes, extent, x, z, symbols, sign_row, sign_const, cos, coupling
):
    """Rotate about ``X^x Z^z``: mix amplitudes i and ``i ^ x`` of every lane."""
    flip = _signs(symbols, lanes, sign_row, sign_const)
    span = 1 << extent
    low = x & -x
    for t in numba.prange(lanes * span):
        lane, i = t >> extent, t & (span - 1)
        if i & low:
            continue
        j = i ^ x
        c = -coupling if flip[lane] else coupling
        a, b = amps[lane, i], amps[lane, j]
        amps[lane, i] = cos * a + c * (-b if _parity(j & z) else b)
        amps[lane, j] = cos * b + c * (-a if _parity(i & z) else a)


@kernel
def _weights(amps, lanes, extent, x, z, i_power):
    """Return each lane's outcome weights ``(w0, w1)`` of measuring ``X^x Z^z``."""
    span = 1 << extent
    w0 = np.zeros(lanes)
    w1 = np.zeros(lanes)
    for lane in numba.prange(lanes):
        for i in range(span):
            a = np.complex128(amps[lane, i])
            if x == 0:
                if _parity(i & z):
                    w1[lane] += a.real**2 + a.imag**2
                else:
                    w0[lane] += a.real**2 + a.imag**2
            else:
                j = i ^ x
                b = i_power * np.complex128(amps[lane, j])
                if _parity(j & z):
                    b = -b
                plus, minus = a + b, a - b
                w0[lane] += (plus.real**2 + plus.imag**2) / 4
                w1[lane] += (minus.real**2 + minus.imag**2) / 4
    return w0, w1


@kernel
def _project(amps, lanes, extent, x, z, pivot, i_power, outcome, weight):
    """Project lane l onto ``outcome[l]`` of weight ``weight[l]``; free the pivot.

    The pivot-1 half of the amplitudes becomes 0: the planner moved the measured
    Pauli onto the pivot, so the pivot ends in ``|0⟩``.
    """
    span = 1 << extent
    bit = 1 << pivot
    for lane in numba.prange(lanes):
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


@kernel
def _shot_symbols(symbols, order, owner, entry_symbols, lane_branches, dormant, seed):
    """Return every shot's symbols: its f-bits, its lane's branches, its coins."""
    out = np.empty_like(symbols)
    num_words = symbols.shape[1]
    for position in numba.prange(len(order)):
        shot, entry = order[position], owner[position]
        for w in range(num_words):
            word = symbols[shot, w] | (entry_symbols[entry, w] & lane_branches[w])
            if dormant[w]:
                n = np.uint64(position * num_words + w)
                _, coin = splitmix64(seed + n * GOLDEN)
                word |= np.uint32(coin & np.uint64(0xFFFFFFFF)) & dormant[w]
            out[shot, w] = word
    return out


class _Lanes:
    """Live lanes in growable arrays: rows ``0 … size-1`` are live.

    Every lane holds at least one shot, so the arrays never need more rows than
    there are shots.
    """

    def __init__(self, symbols, start, count, width: int, dtype: np.dtype):
        self.size = len(start)
        self.shots = max(1, int(np.sum(count)))
        capacity = min(max(16, 2 * self.size), self.shots)
        self.amps = np.zeros((capacity, 1 << width), dtype=dtype)
        self.amps[: self.size, 0] = 1
        self.symbols = np.zeros((capacity, symbols.shape[1]), dtype=np.uint32)
        self.symbols[: self.size] = symbols
        self.start = np.zeros(capacity, dtype=np.int64)
        self.start[: self.size] = start
        self.count = np.zeros(capacity, dtype=np.int64)
        self.count[: self.size] = count

    def split(self, ones: np.ndarray, span: int) -> tuple[np.ndarray, np.ndarray]:
        """Split every lane whose shots see both outcomes.

        Returns each lane's outcome and the lane its amplitudes came from (its
        parent for a new child).
        """
        size, count = self.size, self.count[: self.size]
        outcome = ones == count
        parents = np.flatnonzero((ones > 0) & ~outcome)
        new = size + len(parents)
        if new > len(self.count):
            self._grow(new, span)
        self.amps[size:new, :span] = self.amps[parents, :span]
        self.symbols[size:new] = self.symbols[parents]
        self.count[size:new] = ones[parents]
        self.start[size:new] = self.start[parents] + count[parents] - ones[parents]
        self.count[parents] -= ones[parents]
        self.size = new
        children = np.ones(len(parents), dtype=bool)
        return np.concatenate([outcome, children]), np.concatenate(
            [np.arange(size), parents]
        )

    def prune(self, dies: np.ndarray, span: int) -> tuple[np.ndarray, ...]:
        """Remove the lanes that ``dies`` marks; return their starts, counts, symbols."""
        gone, kept = np.flatnonzero(dies), np.flatnonzero(~dies)
        removed = (self.start[gone], self.count[gone], self.symbols[gone])
        self.amps[: len(kept), :span] = self.amps[kept, :span]
        for array in (self.symbols, self.start, self.count):
            array[: len(kept)] = array[kept]
        self.size = len(kept)
        return removed

    def _grow(self, needed: int, span: int) -> None:
        capacity = min(max(needed, 2 * len(self.count)), self.shots)
        amps = np.zeros((capacity, self.amps.shape[1]), dtype=self.amps.dtype)
        amps[: self.size, :span] = self.amps[: self.size, :span]
        self.amps = amps
        for name in ("symbols", "start", "count"):
            old = getattr(self, name)
            new = np.zeros((capacity, *old.shape[1:]), dtype=old.dtype)
            new[: self.size] = old[: self.size]
            setattr(self, name, new)


def group_on_host(
    words: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort shots by key and return ``(order, start, count)`` of each group.

    Ties are broken at random (a random permutation, then stable sorts), so the
    shots of a group appear in uniformly random order.
    """
    num_shots = words.shape[0]
    order = rng.permutation(num_shots)
    for column in range(words.shape[1]):
        order = order[np.argsort(words[order, column], kind="stable")]
    ordered = words[order]
    boundary = np.ones(num_shots, dtype=bool)
    boundary[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    start = np.flatnonzero(boundary)
    count = np.diff(np.append(start, num_shots))
    return order, start, count


def _run(
    layout: SharingLayout, lanes: _Lanes, rng: np.random.Generator
) -> tuple[float, list[tuple], int]:
    """Run the tape on ``lanes``; return the norm deviation, dead lanes and peak."""
    tape = layout.tape
    dtype = lanes.amps.dtype
    dead: list[tuple] = []
    deviation, peak, region = 0.0, lanes.size, 0
    for a, kind in enumerate(tape.kind):
        if not lanes.size:
            break
        extent = int(tape.extent[a])
        span = 1 << extent
        if extent > region:
            lanes.amps[: lanes.size, 1 << region : span] = 0
        region = extent
        x, z = int(tape.x_mask[a]), int(tape.z_mask[a])
        i_power = dtype.type(_I_POWERS[(x & z).bit_count() & 3])
        size = lanes.size
        work = size * span
        if kind in (ROTATE_DIAG, ROTATE):
            sign = (lanes.symbols, tape.sign_row[a], int(tape.sign_const[a]))
            cos, sin = tape.cos[a], tape.sin[a]
            if kind == ROTATE_DIAG:
                factor = np.array([cos - 1j * sin, cos + 1j * sin], dtype=dtype)
                _rotate_diag(work, lanes.amps, size, extent, z, *sign, factor)
            else:
                coupling = dtype.type(-1j * sin * i_power)
                _rotate_pair(work, lanes.amps, size, extent, x, z, *sign, cos, coupling)
            continue
        if kind == MEASURE_DORMANT:
            if not layout.lane_dormant[a]:
                continue
            w0 = w1 = np.full(size, 0.5)
        else:
            w0, w1 = _weights(work, lanes.amps, size, extent, x, z, i_power)
            deviation = float(np.max(np.abs(w0 + w1 - 1), initial=deviation))
        seed = rng.integers(0, 2**64, dtype=np.uint64)
        ones = _draw_ones(lanes.count[:size], w0, w1, seed)
        outcome, source = lanes.split(ones, span)
        if kind != MEASURE_DORMANT:
            weight = np.where(outcome, w1[source], w0[source])
            _project(
                lanes.size * span,
                lanes.amps,
                lanes.size,
                extent,
                x,
                z,
                int(tape.pivot[a]),
                i_power,
                outcome,
                weight,
            )
        branch = int(tape.branch[a])
        lanes.symbols[: lanes.size, branch >> 5] |= outcome.astype(np.uint32) << (
            branch & 31
        )
        peak = max(peak, lanes.size)
        if layout.pruned[a]:
            values = _row_parities(0, lanes.symbols[: lanes.size], tape.output_row)
            values ^= tape.output_const ^ layout.prune_flip
            dies = np.any(values & layout.prune_outputs[a], axis=1)
            if dies.any():
                dead.append((*lanes.prune(dies, span), a))
    return deviation, dead, peak


def sample_shared_component(
    component: StatevectorComponent,
    layout: SharingLayout,
    f_params: jax.Array | np.ndarray,
    key: jax.Array,
) -> tuple[np.ndarray, jax.Array, float, dict]:
    """Sample one component's outputs with shared trajectories.

    Args:
        component: The compiled component.
        layout: Its sharing layout.
        f_params: Error-basis bits of every shot, shape ``(B, num_f_global)``.
        key: JAX random key.

    Returns:
        ``(samples, next_key, max_norm_deviation, stats)``, where ``samples`` has
        shape ``(B, num_outputs)`` and ``stats`` counts groups and lanes.

    """
    tape = layout.tape
    key, subkey = jax.random.split(key)
    rng = np.random.default_rng(np.asarray(jax.random.bits(subkey, (4,))))
    f = np.asarray(f_params)
    symbols = _pack_symbols(f.size, f, tape.f_selection, component.num_words)
    work = len(symbols) * max(1, layout.key_rows.size)
    keys = _key_words(work, symbols, layout.key_rows)
    order, start, count = group_on_host(keys, rng)
    representative = symbols[order[start]]
    lanes = _Lanes(
        representative, start, count, component.width, np.dtype(component.dtype)
    )
    deviation, dead, peak = _run(layout, lanes, rng)

    num_actions = len(tape.kind)
    live = slice(0, lanes.size)
    entries = [
        (lanes.start[live], lanes.count[live], lanes.symbols[live], num_actions),
        *dead,
    ]
    entry_start = np.concatenate([e[0] for e in entries])
    entry_count = np.concatenate([e[1] for e in entries])
    entry_symbols = np.concatenate([e[2] for e in entries])
    entry_pruned = np.concatenate([np.full(len(e[0]), e[3]) for e in entries])
    # NOTE: the entries' ranges tile the sorted positions, so sorting them by
    # start gives every position's entry.
    by_start = np.argsort(entry_start)
    owner = np.repeat(by_start, entry_count[by_start])
    coin_seed = rng.integers(0, 2**64, dtype=np.uint64)
    shot_symbols = _shot_symbols(
        symbols.size,
        symbols,
        order,
        owner,
        entry_symbols,
        layout.lane_branches,
        layout.shot_dormant,
        coin_seed,
    )
    values = _row_parities(
        shot_symbols.size * len(tape.output_row), shot_symbols, tape.output_row
    )
    values ^= tape.output_const
    pruned_at = np.empty(len(order), dtype=np.int64)
    pruned_at[order] = entry_pruned[owner]
    samples = values & (layout.ready[None, :] <= pruned_at[:, None])
    stats = {
        "groups": len(start),
        "peak lanes": peak,
        "lanes": lanes.size,
        "dead": sum(len(e[0]) for e in dead),
    }
    return samples, key, deviation, stats
