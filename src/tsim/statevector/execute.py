"""JAX executor for statevector components.

``run`` interprets a component's action tape with one ``lax.fori_loop``; each
step dispatches on the action's kind with ``lax.switch``. The loop carry is a
``LaneState``: one lane per shot, each with ``2^W`` amplitudes, its symbols
(f-bits, then branch bits as they are sampled) and its output records.
All randomness goes through ``choose_outcome``, and ``count`` holds the number
of shots a lane stands for (always 1 for now), so that trajectory sharing can
later split lanes without touching the kernels.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from tsim.statevector.program import StatevectorComponent

_I_POWERS = (1, 1j, -1, -1j)
_TINY = 1e-30
_TWO_TO_32 = 2.0**32


class LaneState(NamedTuple):
    """Per-lane state of the executor; the leading axis is the lane.

    Attributes:
        amps: Amplitudes, shape ``(L, 2^W)``.
        symbols: f-bits, then branch bits, packed into uint32 words,
            shape ``(L, num_words)``.
        records: Output bits, plus one scratch column for actions without a
            slot, shape ``(L, num_outputs + 1)``.
        count: Shots each lane stands for, shape ``(L,)``.
        max_norm_deviation: Largest ``|‖ψ‖² - 1|`` seen at a measurement.

    """

    amps: Array
    symbols: Array
    records: Array
    count: Array  # pyright: ignore[reportIncompatibleMethodOverride]
    max_norm_deviation: Array


def choose_outcome(p1: Array, count: Array, key: Array) -> Array:
    """Return each lane's outcome: True with probability ``p1``.

    Two uint32 draws per lane form a uniform 64-bit fraction ``high + low·2^-32``
    (in units of 2^-32), compared with ``t = p1·2^32`` (an exact scaling), so the
    resolution is 2^-64 for any float dtype of ``p1``. ``p1 = 0`` never and
    ``p1 = 1`` always gives True. ``count`` is unused for now; trajectory sharing
    will split it binomially.
    """
    del count
    high, low = jax.random.bits(key, (2, *p1.shape), dtype=jnp.uint32)
    scaled = jnp.clip(p1, 0, 1) * _TWO_TO_32
    whole = jnp.floor(scaled)
    high_bound = whole.astype(jnp.uint32)
    low_bound = jnp.floor((scaled - whole) * _TWO_TO_32).astype(jnp.uint32)
    below = (high < high_bound) | ((high == high_bound) & (low < low_bound))
    return (p1 >= 1) | ((p1 > 0) & below)


def pack_bits(bits: Array, num_words: int) -> Array:
    """Pack boolean columns into uint32 words: column s is bit s % 32 of word s // 32."""
    lanes, num_bits = bits.shape
    padded = jnp.zeros((lanes, num_words * 32), dtype=jnp.uint32)
    padded = padded.at[:, :num_bits].set(bits.astype(jnp.uint32))
    shifts = jnp.arange(32, dtype=jnp.uint32)
    words = padded.reshape(lanes, num_words, 32) << shifts
    return jnp.sum(words, axis=-1, dtype=jnp.uint32)


def initial_state(component: StatevectorComponent, f_bits: Array) -> LaneState:
    """Return lanes in ``|0…0⟩`` with the given f-bits, shape ``(L, num_f)``."""
    lanes = f_bits.shape[0]
    dtype = jnp.dtype(component.dtype)
    amps = jnp.zeros((lanes, 2**component.width), dtype=dtype).at[:, 0].set(1)
    return LaneState(
        amps=amps,
        symbols=pack_bits(f_bits, component.num_words),
        records=jnp.zeros((lanes, component.num_outputs + 1), dtype=jnp.bool_),
        count=jnp.ones(lanes, dtype=jnp.int32),
        max_norm_deviation=jnp.zeros((), dtype=jnp.finfo(dtype).dtype),
    )


def _index(component: StatevectorComponent) -> Array:
    return jnp.arange(2**component.width, dtype=jnp.uint32)


def _parity(values: Array) -> Array:
    return (jax.lax.population_count(values) & 1).astype(jnp.bool_)


def _sign(component: StatevectorComponent, symbols: Array, a: Array) -> Array:
    """Return each lane's sign bit of action ``a``."""
    ones = jnp.sum(jax.lax.population_count(symbols & component.sign_row[a]), axis=1)
    return ((ones & 1) == 1) ^ component.sign_const[a]


def _apply_pauli(amps: Array, index: Array, x: Array, z: Array) -> Array:
    """Return P·ψ lane by lane, for P = i^|x∧z| X^x Z^z."""
    flipped = index ^ x
    quarter_turns = (jax.lax.population_count(x & z) & 3).astype(jnp.int32)
    i_power = jnp.asarray(_I_POWERS, dtype=amps.dtype)[quarter_turns]
    signs = jnp.where(_parity(flipped & z), -1, 1).astype(amps.dtype)
    return i_power * signs * jnp.take(amps, flipped.astype(jnp.int32), axis=1)


def _write(
    component: StatevectorComponent, state: LaneState, a: Array, outcome: Array
) -> LaneState:
    """Record ``outcome ⊕ sign`` in the action's slot and set its branch bit."""
    value = outcome ^ _sign(component, state.symbols, a)
    slot = component.slot[a]
    column = jnp.where(slot >= 0, slot, component.num_outputs)
    records = state.records.at[:, column].set(value)
    branch = component.branch[a]
    word = jnp.maximum(branch, 0) // 32
    bit = jnp.where(
        branch >= 0,
        jnp.uint32(1) << (jnp.maximum(branch, 0) % 32).astype(jnp.uint32),
        jnp.uint32(0),
    )
    flags = jnp.where(outcome, bit, jnp.uint32(0))
    symbols = state.symbols.at[:, word].set(state.symbols[:, word] | flags)
    return state._replace(records=records, symbols=symbols)


def _rotate_diag(
    component: StatevectorComponent, state: LaneState, a: Array, key: Array
) -> LaneState:
    del key
    odd = _parity(_index(component) & component.z_mask[a])
    flip = _sign(component, state.symbols, a)[:, None] ^ odd[None, :]
    factor = component.cos[a] - 1j * component.sin[a] * jnp.where(flip, -1, 1)
    return state._replace(amps=(state.amps * factor).astype(state.amps.dtype))


def _rotate(
    component: StatevectorComponent, state: LaneState, a: Array, key: Array
) -> LaneState:
    del key
    paulied = _apply_pauli(
        state.amps, _index(component), component.x_mask[a], component.z_mask[a]
    )
    negate = jnp.where(_sign(component, state.symbols, a), -1, 1)[:, None]
    amps = component.cos[a] * state.amps - 1j * component.sin[a] * negate * paulied
    return state._replace(amps=amps.astype(state.amps.dtype))


def _squared(values: Array) -> Array:
    return jnp.real(values) ** 2 + jnp.imag(values) ** 2


def _probability_of_one(weight_0: Array, weight_1: Array) -> Array:
    """Return ``w1 / (w0 + w1)`` from the outcome weights, without cancellation."""
    return jnp.clip(weight_1 / jnp.maximum(weight_0 + weight_1, _TINY), 0, 1)


def _measured(state: LaneState, new: Array, norm: Array) -> LaneState:
    deviation = jnp.maximum(state.max_norm_deviation, jnp.max(jnp.abs(norm - 1)))
    return state._replace(
        amps=new.astype(state.amps.dtype),
        max_norm_deviation=deviation.astype(state.max_norm_deviation.dtype),
    )


def _measure(
    component: StatevectorComponent, state: LaneState, a: Array, key: Array
) -> LaneState:
    index = _index(component)
    amps = state.amps
    paulied = _apply_pauli(amps, index, component.x_mask[a], component.z_mask[a])
    plus, minus = amps + paulied, amps - paulied
    weight_0 = jnp.sum(_squared(plus), axis=1) / 4
    weight_1 = jnp.sum(_squared(minus), axis=1) / 4
    p1 = _probability_of_one(weight_0, weight_1)
    outcome = choose_outcome(p1, state.count, jax.random.fold_in(key, a))
    weight = jnp.where(outcome, weight_1, weight_0)
    scale = jax.lax.rsqrt(jnp.maximum(2 * weight, _TINY))[:, None]
    keep = ((index >> component.pivot[a].astype(jnp.uint32)) & 1) == 0
    projected = jnp.where(outcome[:, None], minus, plus)
    new = jnp.where(keep[None, :], projected * scale, 0)
    return _write(component, _measured(state, new, weight_0 + weight_1), a, outcome)


def _measure_diag(
    component: StatevectorComponent, state: LaneState, a: Array, key: Array
) -> LaneState:
    index = _index(component)
    amps = state.amps
    odd = _parity(index & component.z_mask[a])
    weights = _squared(amps)
    weight_0 = jnp.sum(jnp.where(odd[None, :], 0, weights), axis=1)
    weight_1 = jnp.sum(jnp.where(odd[None, :], weights, 0), axis=1)
    p1 = _probability_of_one(weight_0, weight_1)
    outcome = choose_outcome(p1, state.count, jax.random.fold_in(key, a))
    weight = jnp.where(outcome, weight_1, weight_0)
    pivot = component.pivot[a].astype(jnp.uint32)
    bit = jnp.uint32(1) << pivot
    source_if_0 = (index ^ (bit * odd.astype(jnp.uint32))).astype(jnp.int32)
    source_if_1 = (index ^ (bit * (~odd).astype(jnp.uint32))).astype(jnp.int32)
    gathered = jnp.where(
        outcome[:, None],
        jnp.take(amps, source_if_1, axis=1),
        jnp.take(amps, source_if_0, axis=1),
    )
    keep = ((index >> pivot) & 1) == 0
    scale = jax.lax.rsqrt(jnp.maximum(weight, _TINY))[:, None]
    new = jnp.where(keep[None, :], gathered * scale, 0)
    return _write(component, _measured(state, new, weight_0 + weight_1), a, outcome)


def _measure_dormant(
    component: StatevectorComponent, state: LaneState, a: Array, key: Array
) -> LaneState:
    half = jnp.full(state.count.shape, 0.5, dtype=state.max_norm_deviation.dtype)
    outcome = choose_outcome(half, state.count, jax.random.fold_in(key, a))
    return _write(component, state, a, outcome)


def _record(
    component: StatevectorComponent, state: LaneState, a: Array, key: Array
) -> LaneState:
    del key
    return _write(component, state, a, jnp.zeros(state.count.shape, dtype=jnp.bool_))


# NOTE: the order must match the kind codes ROTATE_DIAG … RECORD in program.py.
_KERNELS = (_rotate_diag, _rotate, _measure_diag, _measure, _measure_dormant, _record)


def run(
    component: StatevectorComponent,
    state: LaneState,
    key: Array,
    start: int | Array,
    stop: int | Array,
) -> LaneState:
    """Run actions ``start … stop-1`` of the tape on every lane.

    A component without actions returns ``state`` unchanged.
    """
    if component.num_actions == 0:
        return state
    branches = [partial(kernel, component) for kernel in _KERNELS]

    def step(a: Array, carry: LaneState) -> LaneState:
        return jax.lax.switch(component.kind[a], branches, carry, a, key)

    return jax.lax.fori_loop(start, stop, step, state)


@jax.jit
def _run_lanes(
    component: StatevectorComponent, f_bits: Array, key: Array
) -> tuple[Array, Array]:
    state = run(
        component, initial_state(component, f_bits), key, 0, component.num_actions
    )
    return state.records[:, : component.num_outputs], state.max_norm_deviation


def sample_statevector_component(
    component: StatevectorComponent, f_params: Array, key: Array
) -> tuple[Array, Array, Array]:
    """Sample one component's outputs; the contract of ``sampler.sample_component``.

    Args:
        component: The compiled component.
        f_params: Error-basis bits of every shot, shape ``(B, num_f_global)``.
        key: JAX random key.

    Returns:
        ``(samples, next_key, max_norm_deviation)`` with ``samples`` of shape
        ``(B, num_outputs)``.

    """
    f_bits = f_params[:, component.f_selection].astype(jnp.bool_)
    key, subkey = jax.random.split(key)
    samples, deviation = _run_lanes(component, f_bits, subkey)
    return samples, key, deviation
