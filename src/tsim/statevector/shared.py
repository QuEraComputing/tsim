"""Trajectory-sharing executor for statevector components (investigation prototype).

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
draws how many of its shots see the rarer outcome (``_rare_drawn``, which
resolves the chance of at least one to 2^-64 like ``choose_outcome``) and
splits: the child takes the last n shots of the range. Shots in a lane agree on
the key, so they are exchangeable, and the sort breaks ties at random, so each
range is in uniformly random order: this assignment gives every shot
independent outputs. Pruned lanes move to a dead table that keeps their range,
outcomes and the pruning action but no amplitudes. A dormant measurement splits
lanes the same way (p = 1/2) when its branch reaches a key sign; otherwise it is
a fair coin drawn per shot at the end.

Capacities are static. The dead table holds one entry per shot, so it never
fills. When the lane table is full the loop stops before the action, the host
doubles it and resumes; every lane slot draws from its own key, so the redone
action draws the same numbers. After the tape each shot's outputs are evaluated
from its own f-bits, its lane's outcomes and its own dormant coins; a pruned
shot's outputs that need a branch bit set after its pruning action are False.
"""

from __future__ import annotations

from dataclasses import dataclass
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

_MIN_CAPACITY = 16


class SharingLayout(eqx.Module):
    """Which symbols the shots of a lane must share, from the sign and output rows.

    Attributes:
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
        num_key_words: Number of uint32 words of a packed key.

    """

    key_rows: Array
    lane_dormant: Array
    pruned: Array
    prune_outputs: Array
    prune_flip: Array
    lane_branches: Array
    shot_dormant: Array
    ready: Array
    num_key_words: int = eqx.field(static=True)


@dataclass
class Sharing:
    """A component's sharing layout and the largest lane table it has needed.

    Every new capacity compiles the lane loop again (about a second), so narrow
    components start each batch at ``lane_capacity`` and settle on one compiled
    loop.
    """

    layout: SharingLayout
    lane_capacity: int = 0


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
    num_words = component.num_words
    num_f = len(component.f_selection)
    kind = np.asarray(component.kind)
    branch = np.asarray(component.branch)
    num_actions = len(kind)
    sign_bits = _unpack(np.asarray(component.sign_row), num_words)
    output_bits = _unpack(np.asarray(component.output_row), num_words)
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
    pruned = prune_outputs.any(axis=1)

    rotation = np.isin(kind, (ROTATE_DIAG, ROTATE))
    keyed_bits = np.vstack([sign_bits[rotation], output_bits[postselected]])
    key_bits = keyed_bits[:, :num_f]
    key_bits = key_bits[key_bits.any(axis=1)]
    if len(key_bits):
        key_bits, _ = find_basis(key_bits.astype(np.uint8))
        key_bits = np.asarray(key_bits, dtype=bool)
    num_key_words = max(1, -(-len(key_bits) // 32))

    reaches_key = np.asarray(np.any(keyed_bits, axis=0), dtype=bool)
    dormant = kind == MEASURE_DORMANT
    lane_dormant = dormant & (pruned | ((branch >= 0) & reaches_key[branch]))
    active = np.isin(kind, (MEASURE_DIAG, MEASURE))
    lane_bits = np.zeros(32 * num_words, dtype=bool)
    lane_bits[branch[(active | lane_dormant) & (branch >= 0)]] = True
    shot_bits = np.zeros(32 * num_words, dtype=bool)
    shot_bits[branch[dormant & ~lane_dormant & (branch >= 0)]] = True

    return SharingLayout(
        key_rows=jnp.asarray(_pack(key_bits, num_words).reshape(-1, num_words)),
        lane_dormant=jnp.asarray(lane_dormant),
        pruned=jnp.asarray(pruned),
        prune_outputs=jnp.asarray(prune_outputs),
        prune_flip=jnp.asarray(flipped),
        lane_branches=jnp.asarray(_pack(lane_bits, num_words)),
        shot_dormant=jnp.asarray(_pack(shot_bits, num_words)),
        ready=jnp.asarray(ready.astype(np.int32)),
        num_key_words=num_key_words,
    )


class Lanes(NamedTuple):
    """Lane and dead tables of the shared executor.

    Live lanes occupy slots ``0 … used-1`` of the lane table; dead lanes occupy
    slots ``0 … dead_used-1`` of the dead table, which has one slot per shot.
    """

    amps: Array
    symbols: Array
    count: Array  # pyright: ignore[reportIncompatibleMethodOverride]
    start: Array
    used: Array
    dead_symbols: Array
    dead_count: Array
    dead_start: Array
    dead_action: Array
    dead_used: Array
    action: Array
    full: Array
    max_norm_deviation: Array


def _parity(values: Array) -> Array:
    return (jnp.sum(jax.lax.population_count(values), axis=-1) & 1) == 1


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
    """Sort shots by key and return ``(order, start, count)`` of each group.

    Ties are broken at random (a random permutation, then stable sorts), so the
    shots of a group appear in uniformly random order. numpy's radix sort is an
    order of magnitude faster than XLA's sort on CPU; a GPU build would sort on
    the device.
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


@partial(jax.jit, static_argnames=("capacity",))
def initial_lanes(
    component: StatevectorComponent,
    f_packed: Array,
    order: Array,
    start: Array,
    count: Array,
    num_groups: Array,
    capacity: int,
) -> Lanes:
    """Return one lane per group in ``|0…0⟩``, with an empty dead table."""
    num_shots = f_packed.shape[0]
    padding = max(0, capacity - start.shape[0])
    start = jnp.pad(start, (0, padding), constant_values=num_shots)[:capacity]
    count = jnp.pad(count, (0, padding))[:capacity]
    dtype = jnp.dtype(component.dtype)
    representative = order[jnp.minimum(start, num_shots - 1)]
    symbols = jnp.where(
        (jnp.arange(capacity) < num_groups)[:, None], f_packed[representative], 0
    )
    return Lanes(
        amps=jnp.zeros((capacity, 2**component.width), dtype=dtype).at[:, 0].set(1),
        symbols=symbols.astype(jnp.uint32),
        count=count,
        start=start,
        used=jnp.asarray(num_groups, dtype=jnp.int32),
        dead_symbols=jnp.zeros((num_shots, component.num_words), jnp.uint32),
        dead_count=jnp.zeros(num_shots, dtype=jnp.int32),
        dead_start=jnp.zeros(num_shots, dtype=jnp.int32),
        dead_action=jnp.zeros(num_shots, dtype=jnp.int32),
        dead_used=jnp.zeros((), dtype=jnp.int32),
        action=jnp.zeros((), dtype=jnp.int32),
        full=jnp.zeros((), dtype=jnp.bool_),
        max_norm_deviation=jnp.zeros((), dtype=jnp.finfo(dtype).dtype),
    )


def _lane_state(lanes: Lanes) -> LaneState:
    return LaneState(
        amps=lanes.amps,
        symbols=lanes.symbols,
        count=lanes.count,
        max_norm_deviation=lanes.max_norm_deviation,
    )


def _rare_drawn(key: Array, count: Array, q: Array) -> Array:
    """Return a Binomial(count, q) draw per lane slot, for q ≤ 1/2.

    Whether any shot sees the rare outcome is decided by ``choose_outcome``
    (resolution 2^-64) with P(≥ 1) = 1 - (1 - q)^count; float32 binomial
    samplers cut that chance to a 2^-23 grid. Given at least one, the first such
    shot J follows a truncated geometric law and the remaining count - J shots
    a binomial. Slot s draws from ``fold_in(key, s)``, so a lane's draw does not
    depend on the table size.
    """
    slots = jnp.arange(count.shape[0])
    keys = jax.vmap(lambda s: jax.random.split(jax.random.fold_in(key, s), 3))(slots)
    trials = count.astype(q.dtype)
    log_keep = jnp.log1p(-q)
    p_any = -jnp.expm1(trials * log_keep)
    p_none = jnp.exp(trials * log_keep)
    once = jnp.ones(1, dtype=jnp.int32)
    hit = jax.vmap(
        lambda k, p0, p1: execute.choose_outcome(p0[None], p1[None], once, k)[0]
    )(keys[:, 0], p_none, p_any)

    def after_first_hit() -> Array:
        uniform = jax.vmap(lambda k: jax.random.uniform(k, dtype=q.dtype))(keys[:, 1])
        first = jnp.ceil(
            jnp.log1p(-uniform * p_any) / jnp.where(log_keep < 0, log_keep, -1)
        )
        first = jnp.clip(first, 1, jnp.maximum(trials, 1))
        rest = jax.vmap(jax.random.binomial)(keys[:, 2], trials - first, q)
        return jnp.where(hit, 1 + jnp.round(rest), 0).astype(jnp.int32)

    # NOTE: rare outcomes rarely happen, so most actions skip the binomials.
    rare = jax.lax.cond(
        jnp.any(hit), after_first_hit, lambda: jnp.zeros(count.shape, jnp.int32)
    )
    return jnp.clip(rare, 0, count)


def _ones_drawn(key: Array, count: Array, weight_0: Array, weight_1: Array) -> Array:
    """Return how many of each lane's shots see outcome 1, from the outcome weights."""
    total = jnp.maximum(weight_0 + weight_1, execute._TINY)
    rare_is_one = weight_1 <= weight_0
    q = jnp.clip(jnp.minimum(weight_0, weight_1) / total, 0, 0.5)
    rare = _rare_drawn(key, count, q)
    return jnp.where(rare_is_one, rare, count - rare)


def _split(lanes: Lanes, ones: Array) -> tuple[Lanes, Array, Array]:
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
    component: StatevectorComponent, layout: SharingLayout, lanes: Lanes, a: Array
) -> Lanes:
    """Move live lanes that fail the postselection after action ``a`` to the dead table."""
    capacity = lanes.count.shape[0]
    num_shots = lanes.dead_count.shape[0]
    live = jnp.arange(capacity) < lanes.used
    values = execute.outputs(component, _lane_state(lanes)) ^ layout.prune_flip
    dies = live & jnp.any(values & layout.prune_outputs[a], axis=1)
    dead_slot = jnp.where(dies, lanes.dead_used + jnp.cumsum(dies) - 1, num_shots)
    keeps = live & ~dies
    kept_slot = jnp.where(keeps, jnp.cumsum(keeps) - 1, capacity)

    def compact(values):
        return jnp.zeros_like(values).at[kept_slot].set(values, mode="drop")

    def bury(dead, values):
        return dead.at[dead_slot].set(values, mode="drop")

    return lanes._replace(
        amps=compact(lanes.amps),
        symbols=compact(lanes.symbols),
        count=compact(lanes.count),
        start=compact(lanes.start),
        used=jnp.sum(keeps).astype(jnp.int32),
        dead_symbols=bury(lanes.dead_symbols, lanes.symbols),
        dead_count=bury(lanes.dead_count, lanes.count),
        dead_start=bury(lanes.dead_start, lanes.start),
        dead_action=bury(lanes.dead_action, jnp.full(capacity, a, dtype=jnp.int32)),
        dead_used=(lanes.dead_used + jnp.sum(dies)).astype(jnp.int32),
    )


def _finish(component, layout, lanes, split, a) -> Lanes:
    """Prune ``split`` and advance, or flag ``lanes`` if the lane table is full."""
    pruned = jax.lax.cond(
        layout.pruned[a],
        lambda: _prune(component, layout, split, a),
        lambda: split,
    )
    full = split.used > lanes.count.shape[0]
    advanced = pruned._replace(action=(a + 1).astype(jnp.int32))
    flagged = lanes._replace(full=jnp.asarray(True))
    return jax.tree.map(lambda old, new: jnp.where(full, old, new), flagged, advanced)


def _measure_weights(
    component: StatevectorComponent, amps: Array, a: Array
) -> tuple[Array, Array]:
    """Return each lane's outcome weights ``(w0, w1)`` of ``MEASURE`` action ``a``."""
    paulied = execute._apply_pauli(
        amps, execute._index(component), component.x_mask[a], component.z_mask[a]
    )
    weight_0 = jnp.sum(execute._squared(amps + paulied), axis=1) / 4
    weight_1 = jnp.sum(execute._squared(amps - paulied), axis=1) / 4
    return weight_0, weight_1


def _measure_project(
    component: StatevectorComponent, amps: Array, a: Array, outcome: Array, weight
) -> Array:
    """Return the amplitudes after ``MEASURE`` ``a`` gave ``outcome`` of ``weight``."""
    index = execute._index(component)
    paulied = execute._apply_pauli(
        amps, index, component.x_mask[a], component.z_mask[a]
    )
    projected = jnp.where(outcome[:, None], amps - paulied, amps + paulied)
    scale = jax.lax.rsqrt(jnp.maximum(2 * weight, execute._TINY))[:, None]
    keep = ((index >> component.pivot[a].astype(jnp.uint32)) & 1) == 0
    return jnp.where(keep[None, :], projected * scale, 0)


def _measure_diag_weights(
    component: StatevectorComponent, amps: Array, a: Array
) -> tuple[Array, Array]:
    """Return each lane's outcome weights ``(w0, w1)`` of ``MEASURE_DIAG`` ``a``."""
    odd = execute._parity(execute._index(component) & component.z_mask[a])
    weights = execute._squared(amps)
    weight_0 = jnp.sum(jnp.where(odd[None, :], 0, weights), axis=1)
    weight_1 = jnp.sum(jnp.where(odd[None, :], weights, 0), axis=1)
    return weight_0, weight_1


def _measure_diag_project(
    component: StatevectorComponent, amps: Array, a: Array, outcome: Array, weight
) -> Array:
    """Return the amplitudes after ``MEASURE_DIAG`` ``a`` gave ``outcome``."""
    index = execute._index(component)
    odd = execute._parity(index & component.z_mask[a])
    pivot = component.pivot[a].astype(jnp.uint32)
    partner = jnp.take(amps, (index ^ (jnp.uint32(1) << pivot)).astype(jnp.int32), 1)
    gathered = jnp.where(odd[None, :] == outcome[:, None], amps, partner)
    scale = jax.lax.rsqrt(jnp.maximum(weight, execute._TINY))[:, None]
    keep = ((index >> pivot) & 1) == 0
    return jnp.where(keep[None, :], gathered * scale, 0)


def _measure_kernel(weights, project):
    def kernel(component, layout, lanes, a, key):
        weight_0, weight_1 = weights(component, lanes.amps, a)
        live = jnp.arange(lanes.count.shape[0]) < lanes.used
        deviation = jnp.max(jnp.where(live, jnp.abs(weight_0 + weight_1 - 1), 0))
        ones = _ones_drawn(jax.random.fold_in(key, a), lanes.count, weight_0, weight_1)
        split, outcome, source = _split(lanes, ones)
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
        return _finish(component, layout, lanes, split, a)

    return kernel


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
        split, outcome, source = _split(lanes, ones)
        split = split._replace(
            amps=lanes.amps[source],
            symbols=_set_branch(component, split.symbols, a, outcome),
        )
        return _finish(component, layout, lanes, split, a)

    def skip(lanes):
        return lanes._replace(action=(a + 1).astype(jnp.int32))

    return jax.lax.cond(layout.lane_dormant[a], split_lanes, skip, lanes)


# NOTE: the order must match the kind codes ROTATE_DIAG … MEASURE_DORMANT in
# program.py.
_KERNELS = (
    _rotate_kernel(execute._rotate_diag),
    _rotate_kernel(execute._rotate),
    _measure_kernel(_measure_diag_weights, _measure_diag_project),
    _measure_kernel(_measure_weights, _measure_project),
    _dormant,
)


@jax.jit
def run_lanes(
    component: StatevectorComponent,
    layout: SharingLayout,
    lanes: Lanes,
    key: Array,
    stop: Array,
) -> Lanes:
    """Run actions ``lanes.action … stop-1``, or until the lane table is full."""
    branches = [partial(kernel, component, layout) for kernel in _KERNELS]

    def step(lanes: Lanes) -> Lanes:
        a = lanes.action
        return jax.lax.switch(component.kind[a], branches, lanes, a, key)

    def going(lanes: Lanes) -> Array:
        return (lanes.action < stop) & ~lanes.full

    return jax.lax.while_loop(going, step, lanes)


@partial(jax.jit, static_argnames=("capacity",))
def resize(lanes: Lanes, capacity: int) -> Lanes:
    """Return ``lanes`` with a new lane-table capacity (the live lanes must fit)."""

    def fit(values):
        if capacity <= values.shape[0]:
            return values[:capacity]
        padding = jnp.zeros(
            (capacity - values.shape[0], *values.shape[1:]), values.dtype
        )
        return jnp.concatenate([values, padding])

    return lanes._replace(
        amps=fit(lanes.amps),
        symbols=fit(lanes.symbols),
        count=fit(lanes.count),
        start=fit(lanes.start),
        full=jnp.zeros((), dtype=jnp.bool_),
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
    """Return every shot's outputs, in the original shot order."""
    num_shots = f_packed.shape[0]
    capacity = lanes.count.shape[0]
    live = jnp.concatenate(
        [jnp.arange(capacity) < lanes.used, jnp.arange(num_shots) < lanes.dead_used]
    )
    count = jnp.concatenate([lanes.count, lanes.dead_count])
    start = jnp.where(
        live & (count > 0), jnp.concatenate([lanes.start, lanes.dead_start]), num_shots
    )
    symbols = jnp.concatenate([lanes.symbols, lanes.dead_symbols])
    pruned_at = jnp.concatenate(
        [jnp.full(capacity, component.num_actions, jnp.int32), lanes.dead_action]
    )
    # The ranges partition the positions, so each position's entry is the one
    # starting at the latest range start at or before it.
    position = jnp.arange(num_shots)
    starting = (
        jnp.full(num_shots, -1, jnp.int32)
        .at[start]
        .set(jnp.arange(start.shape[0], dtype=jnp.int32), mode="drop")
    )
    latest_start = jax.lax.cummax(jnp.where(starting >= 0, position, -1))
    entry = starting[latest_start]
    coins = jax.random.bits(key, (num_shots, component.num_words), dtype=jnp.uint32)
    shot_symbols = (
        f_packed[order]
        | (symbols[entry] & layout.lane_branches)
        | (coins & layout.shot_dormant)
    )
    values = execute._affine(
        shot_symbols[:, None, :], component.output_row, component.output_const
    )
    written = layout.ready[None, :] <= pruned_at[entry][:, None]
    values = values & written
    return jnp.zeros_like(values).at[order].set(values)


def _capacity(needed: int) -> int:
    return max(_MIN_CAPACITY, 1 << int(np.ceil(np.log2(max(needed, 1)))))


@partial(jax.jit, static_argnames=("num_words",))
def _packed_f(f_params: Array, f_selection: Array, num_words: int) -> Array:
    return pack_bits(f_params[:, f_selection].astype(jnp.bool_), num_words)


def sample_shared_component(
    component: StatevectorComponent,
    sharing: Sharing,
    f_params: Array,
    key: Array,
    *,
    segment: int | None = None,
) -> tuple[Array, Array, Array, dict]:
    """Sample one component's outputs with shared trajectories.

    Args:
        component: The compiled component.
        sharing: Its sharing layout and capacity hint (updated in place).
        f_params: Error-basis bits of every shot, shape ``(B, num_f_global)``.
        key: JAX random key.
        segment: Actions between host checks that shrink the lane table to the
            live lanes. Defaults to 32 for ``W ≥ 8`` and to the whole tape below,
            where shrinking saves less array work than its compilations cost.

    Returns:
        ``(samples, next_key, max_norm_deviation, stats)``, where ``samples`` has
        shape ``(B, num_outputs)`` and ``stats`` counts groups and lanes.

    """
    layout = sharing.layout
    num_actions = component.num_actions
    if segment is None:
        segment = 32 if component.width >= 8 else max(num_actions, 1)
    f_packed = _packed_f(f_params, component.f_selection, component.num_words)
    num_shots = f_packed.shape[0]
    key, order_key, lane_key, coin_key = jax.random.split(key, 4)
    rng = np.random.default_rng(np.asarray(jax.random.bits(order_key, (4,))))
    order_np, start_np, count_np = group_on_host(
        np.asarray(key_words(layout, f_packed)), rng
    )
    groups = len(start_np)
    sticky = sharing.lane_capacity if segment >= num_actions else 0
    capacity = max(_capacity(2 * groups), sticky)
    start_padded = np.full(capacity, num_shots, dtype=np.int32)
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
    )
    resumes = peak = dead_used = 0
    largest = capacity
    action, used = 0, groups
    while action < num_actions:
        stop = jnp.asarray(min(action + segment, num_actions), dtype=jnp.int32)
        lanes = run_lanes(component, layout, lanes, lane_key, stop)
        full, action, used, dead_used = (
            int(value)
            for value in jax.device_get(
                (lanes.full, lanes.action, lanes.used, lanes.dead_used)
            )
        )
        peak = max(peak, used)
        capacity = lanes.count.shape[0]
        if full:
            resumes += 1
            lanes = resize(lanes, capacity=2 * capacity)
        elif action < num_actions and _capacity(2 * used) < capacity:
            lanes = resize(lanes, capacity=_capacity(2 * used))
        largest = max(largest, lanes.count.shape[0])
    sharing.lane_capacity = max(sharing.lane_capacity, largest)
    samples = reconstruct(component, layout, order, lanes, f_packed, coin_key)
    stats = {
        "groups": groups,
        "peak lanes": peak,
        "lanes": used,
        "dead": dead_used,
        "lane_capacity": lanes.count.shape[0],
        "resumes": resumes,
    }
    return samples, key, lanes.max_norm_deviation, stats
