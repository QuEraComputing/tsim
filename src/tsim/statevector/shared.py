"""Trajectory-sharing executor for statevector components (investigation prototype).

Shots that agree on every symbol that reaches the amplitudes share one lane. The
only per-shot input to the amplitudes is the sign of a rotation (a measurement's
sign only flips its record), so a shot's grouping key is the f-bit part of every
rotation sign. With postselection the key also holds the f-bit part of every
postselected output's sign, so that a lane is pruned as a whole as soon as one
of its postselected records is 1.

A lane holds amplitudes, the symbols of one of its shots (its f-bits agree with
every member on the key, and its branch bits are the lane's outcomes), a shot
count and a range in the key-sorted order of the shots. At a measurement the lane
draws how many of its shots see each outcome, binomially and for the rarer
outcome, so that rounding p₁ to 1 cannot hide it, and splits: the child takes the
last n shots of the range. Shots in a lane agree on the key, so they are
exchangeable, and the sort breaks ties at random, so each range is in uniformly
random order: this assignment gives every shot independent outputs.
Pruned lanes move to a dead table that keeps their range and outcomes but no
amplitudes. A dormant measurement splits lanes the same way (p = 1/2) when its
branch reaches a key sign; otherwise it is a fair coin drawn per shot at the end.

Capacities are static. When a table is full the loop stops before the action,
the host doubles that table and resumes. After the tape each shot's records are
evaluated from its own f-bits, its lane's outcomes and its own dormant coins.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from tsim.statevector import execute
from tsim.statevector.execute import LaneState, pack_bits
from tsim.statevector.program import (
    MEASURE,
    MEASURE_DIAG,
    MEASURE_DORMANT,
    ROTATE,
    ROTATE_DIAG,
    StatevectorComponent,
)
from tsim.utils.linalg import find_basis

LANES_FULL = 1
DEAD_FULL = 2
_MIN_CAPACITY = 16


class SharingLayout(eqx.Module):
    """Which symbols the shots of a lane must share, from the sign rows.

    Attributes:
        key_rows: f-bit masks whose parities form the grouping key, a basis of the
            f-parts of every rotation sign and every postselected sign,
            shape ``(K, num_words)``.
        lane_dormant: Dormant measurements that split lanes, shape ``(A,)``.
        pruned: Actions whose record is postselected, shape ``(A,)``.
        lane_branches: Branch bits a lane carries, shape ``(num_words,)``.
        shot_dormant: Branch bits of the dormant measurements drawn per shot,
            shape ``(num_words,)``.
        slot_action: The action that writes each output slot, shape
            ``(num_outputs,)``.
        num_key_words: Number of uint32 words of a packed key.

    """

    key_rows: Array
    lane_dormant: Array
    pruned: Array
    lane_branches: Array
    shot_dormant: Array
    slot_action: Array
    num_key_words: int = eqx.field(static=True)


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
    component: StatevectorComponent, postselected: np.ndarray | None = None
) -> SharingLayout:
    """Return the sharing layout of a component.

    Args:
        component: The compiled component.
        postselected: Whether each output slot is postselected (its record must
            be 0), shape ``(num_outputs,)``. ``None`` prunes nothing.

    Raises:
        ValueError: If a recorded measurement keeps no branch bit (its outcome
            could not be recovered per shot).

    """
    num_words = component.num_words
    num_f = len(component.f_selection)
    kind = np.asarray(component.kind)
    slot = np.asarray(component.slot)
    branch = np.asarray(component.branch)
    bits = _unpack(np.asarray(component.sign_row), num_words)
    if postselected is None:
        postselected = np.zeros(component.num_outputs, dtype=bool)
    postselected = np.asarray(postselected, dtype=bool)

    measurement = np.isin(kind, (MEASURE_DIAG, MEASURE, MEASURE_DORMANT))
    if np.any(measurement & (slot >= 0) & (branch < 0)):
        raise ValueError("a recorded measurement has no branch bit")
    pruned = (slot >= 0) & postselected[np.maximum(slot, 0)]
    keyed = np.isin(kind, (ROTATE_DIAG, ROTATE)) | pruned

    key_bits = bits[keyed][:, :num_f]
    key_bits = key_bits[key_bits.any(axis=1)]
    if len(key_bits):
        key_bits, _ = find_basis(key_bits.astype(np.uint8))
        key_bits = np.asarray(key_bits, dtype=bool)
    num_key_words = max(1, -(-len(key_bits) // 32))

    reaches_key = np.asarray(np.any(bits[keyed], axis=0), dtype=bool)
    dormant = kind == MEASURE_DORMANT
    lane_dormant = dormant & (pruned | ((branch >= 0) & reaches_key[branch]))
    active = np.isin(kind, (MEASURE_DIAG, MEASURE))
    lane_bits = np.zeros(32 * num_words, dtype=bool)
    lane_bits[branch[(active | lane_dormant) & (branch >= 0)]] = True
    shot_bits = np.zeros(32 * num_words, dtype=bool)
    shot_bits[branch[dormant & ~lane_dormant & (branch >= 0)]] = True

    slot_action = np.full(component.num_outputs, -1, dtype=np.int32)
    for a in np.flatnonzero(slot >= 0):
        slot_action[slot[a]] = a

    return SharingLayout(
        key_rows=jnp.asarray(_pack(key_bits, num_words).reshape(-1, num_words)),
        lane_dormant=jnp.asarray(lane_dormant),
        pruned=jnp.asarray(pruned),
        lane_branches=jnp.asarray(_pack(lane_bits, num_words)),
        shot_dormant=jnp.asarray(_pack(shot_bits, num_words)),
        slot_action=jnp.asarray(slot_action),
        num_key_words=num_key_words,
    )


class Lanes(NamedTuple):
    """Lane and dead tables of the shared executor.

    Live lanes occupy slots ``0 … used-1`` of the lane table; dead lanes occupy
    slots ``0 … dead_used-1`` of the dead table.
    """

    amps: Array
    symbols: Array
    count: Array  # pyright: ignore[reportIncompatibleMethodOverride]
    start: Array
    used: Array
    dead_symbols: Array
    dead_count: Array
    dead_start: Array
    dead_used: Array
    action: Array
    overflow: Array
    max_norm_deviation: Array


def _parity(values: Array) -> Array:
    return (jnp.sum(jax.lax.population_count(values), axis=-1) & 1) == 1


@partial(jax.jit, static_argnames=("capacity",))
def group(
    layout: SharingLayout, f_packed: Array, key: Array, capacity: int
) -> tuple[Array, Array, Array, Array]:
    """Sort shots by key and return ``(order, start, count, num_groups)``.

    Ties are broken at random, so the shots of a group appear in a uniformly
    random order and handing a split's outcome to the last shots of a range
    keeps every shot's outputs independent. ``start`` and ``count`` have
    ``capacity`` entries; groups beyond it are lost, so callers check
    ``num_groups <= capacity``.
    """
    num_shots = f_packed.shape[0]
    num_keys = layout.key_rows.shape[0]
    tie_break = jax.random.bits(key, (num_shots,), dtype=jnp.uint32)
    if num_keys == 0:
        order = jnp.argsort(tie_break).astype(jnp.int32)
        boundary = jnp.zeros(num_shots, dtype=jnp.bool_).at[0].set(True)
    else:

        def key_bit(k, words):
            bit = _parity(f_packed & layout.key_rows[k]).astype(jnp.uint32)
            word = k // 32
            shift = (k % 32).astype(jnp.uint32)
            return words.at[:, word].set(words[:, word] | (bit << shift))

        words = jax.lax.fori_loop(
            0,
            num_keys,
            key_bit,
            jnp.zeros((num_shots, layout.num_key_words), dtype=jnp.uint32),
        )
        order = jnp.lexsort((tie_break, *words.T)).astype(jnp.int32)
        ordered = words[order]
        changed = jnp.any(ordered[1:] != ordered[:-1], axis=1)
        boundary = jnp.concatenate([jnp.ones(1, dtype=jnp.bool_), changed])
    start = jnp.nonzero(boundary, size=capacity, fill_value=num_shots)[0]
    following = jnp.concatenate([start[1:], jnp.full(1, num_shots)])
    following = jnp.minimum(following, num_shots)
    count = jnp.where(start < num_shots, following - start, 0).astype(jnp.int32)
    return order, start.astype(jnp.int32), count, jnp.sum(boundary)


@jax.jit
def key_words(layout: SharingLayout, f_packed: Array) -> Array:
    """Return each shot's packed key, shape ``(B, num_key_words)``."""
    num_shots = f_packed.shape[0]
    if layout.key_rows.shape[0] == 0:
        return jnp.zeros((num_shots, layout.num_key_words), dtype=jnp.uint32)

    def key_bit(k, words):
        bit = _parity(f_packed & layout.key_rows[k]).astype(jnp.uint32)
        word = k // 32
        shift = (k % 32).astype(jnp.uint32)
        return words.at[:, word].set(words[:, word] | (bit << shift))

    return jax.lax.fori_loop(
        0,
        layout.key_rows.shape[0],
        key_bit,
        jnp.zeros((num_shots, layout.num_key_words), dtype=jnp.uint32),
    )


def group_on_host(
    words: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort shots by key on the host and return ``(order, start, count)``.

    The same contract as ``group`` (ties broken at random), with numpy's radix
    sort, which is an order of magnitude faster than XLA's sort on CPU.
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
    return order.astype(np.int32), start.astype(np.int32), count.astype(np.int32)


@partial(jax.jit, static_argnames=("capacity", "dead_capacity"))
def initial_lanes(
    component: StatevectorComponent,
    f_packed: Array,
    order: Array,
    start: Array,
    count: Array,
    num_groups: Array,
    capacity: int,
    dead_capacity: int,
) -> Lanes:
    """Return one lane per group in ``|0…0⟩``, with an empty dead table."""
    num_shots = f_packed.shape[0]
    padding = max(0, capacity - start.shape[0])
    start = jnp.pad(start, (0, padding), constant_values=num_shots)[:capacity]
    count = jnp.pad(count, (0, padding))[:capacity]
    dtype = jnp.dtype(component.dtype)
    representative = order[jnp.minimum(start, f_packed.shape[0] - 1)]
    symbols = jnp.where(
        (jnp.arange(capacity) < num_groups)[:, None], f_packed[representative], 0
    )
    return Lanes(
        amps=jnp.zeros((capacity, 2**component.width), dtype=dtype).at[:, 0].set(1),
        symbols=symbols.astype(jnp.uint32),
        count=count,
        start=start,
        used=jnp.asarray(num_groups, dtype=jnp.int32),
        dead_symbols=jnp.zeros((dead_capacity, component.num_words), jnp.uint32),
        dead_count=jnp.zeros(dead_capacity, dtype=jnp.int32),
        dead_start=jnp.zeros(dead_capacity, dtype=jnp.int32),
        dead_used=jnp.zeros((), dtype=jnp.int32),
        action=jnp.zeros((), dtype=jnp.int32),
        overflow=jnp.zeros((), dtype=jnp.int32),
        max_norm_deviation=jnp.zeros((), dtype=jnp.finfo(dtype).dtype),
    )


def _lane_state(lanes: Lanes) -> LaneState:
    return LaneState(
        amps=lanes.amps,
        symbols=lanes.symbols,
        records=jnp.zeros((lanes.count.shape[0], 1), dtype=jnp.bool_),
        count=lanes.count,
        max_norm_deviation=lanes.max_norm_deviation,
    )


def _ones_drawn(key: Array, count: Array, weight_0: Array, weight_1: Array) -> Array:
    """Return how many of each lane's shots see outcome 1, from the outcome weights."""
    total = jnp.maximum(weight_0 + weight_1, execute._TINY)
    rare_is_one = weight_1 <= weight_0
    p_rare = jnp.clip(jnp.minimum(weight_0, weight_1) / total, 0, 1)
    rare = jax.random.binomial(key, count.astype(p_rare.dtype), p_rare)
    rare = jnp.clip(jnp.round(rare).astype(jnp.int32), 0, count)
    return jnp.where(rare_is_one, rare, count - rare)


def _split(lanes: Lanes, a: Array, ones: Array) -> tuple[Lanes, Array, Array]:
    """Split every live lane whose shots see both outcomes.

    Returns the split lanes, each slot's outcome and the source slot of each
    slot's amplitudes (its parent for a new child).
    """
    capacity = lanes.count.shape[0]
    slots = jnp.arange(capacity)
    live = slots < lanes.used
    ones = jnp.where(live, ones, 0)
    split = live & (ones > 0) & (ones < lanes.count)
    child = lanes.used + jnp.cumsum(split) - 1
    target = jnp.where(split, child, capacity)
    source = slots.at[target].set(slots, mode="drop")
    is_child = jnp.zeros(capacity, dtype=jnp.bool_).at[target].set(True, mode="drop")
    outcome = is_child | (live & (ones == lanes.count) & (lanes.count > 0))
    parent_count = lanes.count[source]
    parent_ones = ones[source]
    count = jnp.where(
        is_child, parent_ones, jnp.where(split, lanes.count - ones, lanes.count)
    )
    start = jnp.where(
        is_child, lanes.start[source] + parent_count - parent_ones, lanes.start
    )
    split_lanes = lanes._replace(
        symbols=lanes.symbols[source],
        count=count.astype(jnp.int32),
        start=start.astype(jnp.int32),
        used=(lanes.used + jnp.sum(split)).astype(jnp.int32),
    )
    return split_lanes, outcome, source


def _set_branch(component: StatevectorComponent, symbols, a, outcome):
    branch = component.branch[a]
    word = jnp.maximum(branch, 0) // 32
    bit = jnp.where(
        branch >= 0,
        jnp.uint32(1) << (jnp.maximum(branch, 0) % 32).astype(jnp.uint32),
        jnp.uint32(0),
    )
    flags = jnp.where(outcome, bit, jnp.uint32(0))
    return symbols.at[:, word].set(symbols[:, word] | flags)


def _prune(
    component: StatevectorComponent, lanes: Lanes, a: Array, outcome: Array
) -> Lanes:
    """Move live lanes whose record of action ``a`` is 1 to the dead table.

    The dead table may overflow (``dead_used`` beyond its capacity); the caller
    checks and discards the result.
    """
    capacity = lanes.count.shape[0]
    dead_capacity = lanes.dead_count.shape[0]
    live = jnp.arange(capacity) < lanes.used
    sign = _parity(lanes.symbols & component.sign_row[a]) ^ component.sign_const[a]
    dies = live & (outcome ^ sign)
    dead_slot = jnp.where(dies, lanes.dead_used + jnp.cumsum(dies) - 1, dead_capacity)
    keeps = live & ~dies
    kept_slot = jnp.where(keeps, jnp.cumsum(keeps) - 1, capacity)

    def compact(values):
        return jnp.zeros_like(values).at[kept_slot].set(values, mode="drop")

    return lanes._replace(
        amps=compact(lanes.amps),
        symbols=compact(lanes.symbols),
        count=compact(lanes.count),
        start=compact(lanes.start),
        used=jnp.sum(keeps).astype(jnp.int32),
        dead_symbols=lanes.dead_symbols.at[dead_slot].set(lanes.symbols, mode="drop"),
        dead_count=lanes.dead_count.at[dead_slot].set(lanes.count, mode="drop"),
        dead_start=lanes.dead_start.at[dead_slot].set(lanes.start, mode="drop"),
        dead_used=(lanes.dead_used + jnp.sum(dies)).astype(jnp.int32),
    )


def _measure_kernel(weights, project):
    def kernel(component, layout, lanes, a, key):
        weight_0, weight_1 = weights(component, lanes.amps, a)
        live = jnp.arange(lanes.count.shape[0]) < lanes.used
        deviation = jnp.max(jnp.where(live, jnp.abs(weight_0 + weight_1 - 1), 0))
        ones = _ones_drawn(jax.random.fold_in(key, a), lanes.count, weight_0, weight_1)
        split, outcome, source = _split(lanes, a, ones)
        amps = lanes.amps[source]
        weight = jnp.where(outcome, weight_1[source], weight_0[source])
        split = split._replace(
            amps=project(component, amps, a, outcome, weight).astype(amps.dtype),
            symbols=_set_branch(component, split.symbols, a, outcome),
            max_norm_deviation=jnp.maximum(
                lanes.max_norm_deviation,
                deviation.astype(lanes.max_norm_deviation.dtype),
            ),
        )
        return _finish(component, layout, lanes, split, a, outcome)

    return kernel


def _finish(component, layout, lanes, split, a, outcome) -> Lanes:
    """Prune ``split`` and advance, or return ``lanes`` flagged if a table is full."""
    pruned = jax.lax.cond(
        layout.pruned[a],
        lambda: _prune(component, split, a, outcome),
        lambda: split,
    )
    overflow = jnp.where(
        split.used > lanes.count.shape[0],
        LANES_FULL,
        jnp.where(pruned.dead_used > lanes.dead_count.shape[0], DEAD_FULL, 0),
    ).astype(jnp.int32)
    advanced = pruned._replace(action=(a + 1).astype(jnp.int32))
    flagged = lanes._replace(overflow=overflow)
    return jax.tree.map(
        lambda old, new: jnp.where(overflow != 0, old, new), flagged, advanced
    )


def _rotate_kernel(kernel):
    def apply(component, layout, lanes, a, key):
        del layout
        amps = kernel(component, _lane_state(lanes), a, key).amps
        return lanes._replace(amps=amps, action=(a + 1).astype(jnp.int32))

    return apply


def _dormant(component, layout, lanes, a, key):
    def split_lanes(lanes):
        half = jnp.full(lanes.count.shape, 0.5, dtype=lanes.max_norm_deviation.dtype)
        ones = _ones_drawn(jax.random.fold_in(key, a), lanes.count, half, half)
        split, outcome, source = _split(lanes, a, ones)
        split = split._replace(
            amps=lanes.amps[source],
            symbols=_set_branch(component, split.symbols, a, outcome),
        )
        return _finish(component, layout, lanes, split, a, outcome)

    def skip(lanes):
        return lanes._replace(action=(a + 1).astype(jnp.int32))

    return jax.lax.cond(layout.lane_dormant[a], split_lanes, skip, lanes)


def _record(component, layout, lanes, a, key):
    del key
    outcome = jnp.zeros(lanes.count.shape, dtype=jnp.bool_)
    return _finish(component, layout, lanes, lanes, a, outcome)


# NOTE: the order must match the kind codes ROTATE_DIAG … RECORD in program.py.
_KERNELS = (
    _rotate_kernel(execute._rotate_diag),
    _rotate_kernel(execute._rotate),
    _measure_kernel(execute.measure_diag_weights, execute.measure_diag_project),
    _measure_kernel(execute.measure_weights, execute.measure_project),
    _dormant,
    _record,
)


@jax.jit
def run_lanes(
    component: StatevectorComponent,
    layout: SharingLayout,
    lanes: Lanes,
    key: Array,
    stop: Array,
) -> Lanes:
    """Run actions ``lanes.action … stop-1``, or until a table is full."""
    branches = [partial(kernel, component, layout) for kernel in _KERNELS]

    def step(lanes: Lanes) -> Lanes:
        a = lanes.action
        return jax.lax.switch(component.kind[a], branches, lanes, a, key)

    def going(lanes: Lanes) -> Array:
        return (lanes.action < stop) & (lanes.overflow == 0)

    return jax.lax.while_loop(going, step, lanes)


@partial(jax.jit, static_argnames=("capacity", "dead_capacity"))
def resize(lanes: Lanes, capacity: int, dead_capacity: int) -> Lanes:
    """Return ``lanes`` with new table capacities (live lanes must fit)."""

    def fit(values, size):
        if size <= values.shape[0]:
            return values[:size]
        padding = jnp.zeros((size - values.shape[0], *values.shape[1:]), values.dtype)
        return jnp.concatenate([values, padding])

    return lanes._replace(
        amps=fit(lanes.amps, capacity),
        symbols=fit(lanes.symbols, capacity),
        count=fit(lanes.count, capacity),
        start=fit(lanes.start, capacity),
        dead_symbols=fit(lanes.dead_symbols, dead_capacity),
        dead_count=fit(lanes.dead_count, dead_capacity),
        dead_start=fit(lanes.dead_start, dead_capacity),
        overflow=jnp.zeros((), dtype=jnp.int32),
    )


@jax.jit
def reconstruct(
    component: StatevectorComponent,
    layout: SharingLayout,
    order: Array,
    lanes: Lanes,
    f_packed: Array,
    key: Array,
) -> Array:
    """Return every shot's records, in the original shot order."""
    num_shots = f_packed.shape[0]
    capacity = lanes.count.shape[0]
    dead_capacity = lanes.dead_count.shape[0]
    live = jnp.concatenate(
        [
            jnp.arange(capacity) < lanes.used,
            jnp.arange(dead_capacity) < lanes.dead_used,
        ]
    )
    count = jnp.concatenate([lanes.count, lanes.dead_count])
    start = jnp.where(
        live & (count > 0), jnp.concatenate([lanes.start, lanes.dead_start]), num_shots
    )
    symbols = jnp.concatenate([lanes.symbols, lanes.dead_symbols])
    by_start = jnp.argsort(start)
    position = jnp.arange(num_shots)
    entry = by_start[jnp.searchsorted(start[by_start], position, side="right") - 1]
    coins = jax.random.bits(key, (num_shots, component.num_words), dtype=jnp.uint32)
    shot_symbols = (
        f_packed[order]
        | (symbols[entry] & layout.lane_branches)
        | (coins & layout.shot_dormant)
    )
    actions = jnp.maximum(layout.slot_action, 0)
    ones = jnp.sum(
        jax.lax.population_count(
            shot_symbols[:, None, :] & component.sign_row[actions][None]
        ),
        axis=2,
    )
    sign = ((ones & 1) == 1) ^ component.sign_const[actions]
    branch = component.branch[actions]
    words = jnp.take_along_axis(
        shot_symbols, jnp.broadcast_to(jnp.maximum(branch, 0) // 32, sign.shape), axis=1
    )
    shift = (jnp.maximum(branch, 0) % 32).astype(jnp.uint32)
    outcome = ((words >> shift) & 1).astype(jnp.bool_) & (branch >= 0)
    values = (sign ^ outcome) & (layout.slot_action >= 0)
    return jnp.zeros_like(values).at[order].set(values)


def _capacity(needed: int) -> int:
    return max(_MIN_CAPACITY, 1 << int(np.ceil(np.log2(max(needed, 1)))))


# NOTE: prototype state: the dead-table capacity each layout ended its last batch
# with, so that the next batch starts there instead of growing again.
_DEAD_CAPACITY: dict[int, int] = {}


@partial(jax.jit, static_argnames=("num_words",))
def _packed_f(f_params: Array, f_selection: Array, num_words: int) -> Array:
    return pack_bits(f_params[:, f_selection].astype(jnp.bool_), num_words)


def sample_shared_component(
    component: StatevectorComponent,
    layout: SharingLayout,
    f_params: Array,
    key: Array,
    *,
    segment: int | None = None,
) -> tuple[Array, Array, Array, dict]:
    """Sample one component's outputs with shared trajectories.

    Args:
        component: The compiled component.
        layout: Its sharing layout (``sharing_layout``).
        f_params: Error-basis bits of every shot, shape ``(B, num_f_global)``.
        key: JAX random key.
        segment: Actions between host checks that shrink the lane table to the
            live lanes (default 32; 16 to 64 measured about equally fast).

    Returns:
        ``(samples, next_key, max_norm_deviation, stats)``, where ``samples`` has
        shape ``(B, num_outputs)`` and ``stats`` counts groups and lanes.

    """
    num_actions = component.num_actions
    if segment is None:
        segment = 32
    f_packed = _packed_f(f_params, component.f_selection, component.num_words)
    key, order_key, lane_key, coin_key = jax.random.split(key, 4)
    rng = np.random.default_rng(np.asarray(jax.random.bits(order_key, (4,))))
    order_np, start_np, count_np = group_on_host(
        np.asarray(key_words(layout, f_packed)), rng
    )
    groups = len(start_np)
    capacity = _capacity(2 * groups)
    start_padded = np.full(capacity, len(order_np), dtype=np.int32)
    count_padded = np.zeros(capacity, dtype=np.int32)
    start_padded[:groups], count_padded[:groups] = start_np, count_np
    order = jnp.asarray(order_np)
    lanes = initial_lanes(
        component,
        f_packed,
        order,
        jnp.asarray(start_padded),
        jnp.asarray(count_padded),
        jnp.asarray(groups, dtype=jnp.int32),
        capacity=capacity,
        dead_capacity=_DEAD_CAPACITY.get(id(layout), _MIN_CAPACITY),
    )
    resumes = peak = dead_used = 0
    action, used = 0, groups
    while action < num_actions:
        stop = jnp.asarray(min(action + segment, num_actions), dtype=jnp.int32)
        lanes = run_lanes(component, layout, lanes, lane_key, stop)
        overflow, action, used, dead_used = (
            int(value)
            for value in jax.device_get(
                (lanes.overflow, lanes.action, lanes.used, lanes.dead_used)
            )
        )
        peak = max(peak, used)
        capacity, dead_capacity = lanes.count.shape[0], lanes.dead_count.shape[0]
        if overflow:
            resumes += 1
            capacity *= 2 if overflow == LANES_FULL else 1
            dead_capacity *= 2 if overflow == DEAD_FULL else 1
        elif action < num_actions and _capacity(2 * used) < capacity:
            capacity = _capacity(2 * used)
        if (capacity, dead_capacity) != (
            lanes.count.shape[0],
            lanes.dead_count.shape[0],
        ):
            lanes = resize(lanes, capacity=capacity, dead_capacity=dead_capacity)
    _DEAD_CAPACITY[id(layout)] = lanes.dead_count.shape[0]
    samples = reconstruct(component, layout, order, lanes, f_packed, coin_key)
    stats = {
        "groups": groups,
        "peak lanes": peak,
        "lanes": used,
        "dead": dead_used,
        "lane_capacity": lanes.count.shape[0],
        "dead_capacity": lanes.dead_count.shape[0],
        "resumes": resumes,
    }
    return samples, key, lanes.max_norm_deviation, stats
