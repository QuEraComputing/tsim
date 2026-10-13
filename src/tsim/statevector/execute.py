"""JAX executor for statevector components.

``run`` interprets a component's action tape with one ``lax.fori_loop``; each
step dispatches on the action's kind with ``lax.switch``. The loop carry is a
``LaneState``: one lane per shot, each with ``2^W`` amplitudes and its symbols
(f-bits, then branch bits as they are sampled), from which ``outputs`` reads
the output bits once the tape has run. All randomness goes through
``choose_outcome``.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from tsim.statevector.program import (
    MEASURE,
    MEASURE_DIAG,
    MEASURE_DORMANT,
    ROTATE,
    ROTATE_DIAG,
    StatevectorComponent,
)

_I_POWERS = (1, 1j, -1, -1j)
_TINY = 1e-30
_TWO_TO_32 = 2.0**32


class LaneState(NamedTuple):
    """Per-lane state of the executor; the leading axis is the lane.

    Attributes:
        amps: Amplitudes, shape ``(L, 2^W)``.
        symbols: f-bits, then branch bits, packed into uint32 words,
            shape ``(L, num_words)``.
        max_norm_deviation: Largest ``|‖ψ‖² - 1|`` seen at a measurement.

    """

    amps: Array
    symbols: Array
    max_norm_deviation: Array


def lane_bytes(component: StatevectorComponent) -> int:
    """Return the peak device memory of one lane, temporary amplitudes included."""
    # NOTE: XLA's memory analysis of the executor shows four amplitude arrays.
    amplitudes = 4 * 2**component.width * jnp.dtype(component.dtype).itemsize
    return amplitudes + 4 * component.num_words + component.num_outputs


def choose_outcome(p0: Array, p1: Array, key: Array) -> Array:
    """Return each lane's outcome: True with probability ``p1 = 1 - p0``.

    The less likely outcome is drawn, so that both outcomes keep the relative
    precision of their probability: in float32, ``p1`` alone rounds to 1 once
    ``p0`` is below about 6·10⁻⁸. Two uint32 draws per lane form a uniform 64-bit
    fraction, compared with ``p·2^32`` (an exact scaling) for the smaller ``p``,
    so the resolution is 2^-64 for any float dtype, and an outcome of
    probability 0 is never drawn.
    """
    high, low = jax.random.bits(key, (2, *p1.shape), dtype=jnp.uint32)
    one_is_rare = p1 <= p0
    scaled = jnp.clip(jnp.minimum(p0, p1), 0, 0.5) * _TWO_TO_32
    whole = jnp.floor(scaled)
    high_bound = whole.astype(jnp.uint32)
    low_bound = jnp.floor((scaled - whole) * _TWO_TO_32).astype(jnp.uint32)
    rare = (high < high_bound) | ((high == high_bound) & (low < low_bound))
    return rare == one_is_rare


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
        max_norm_deviation=jnp.zeros((), dtype=jnp.finfo(dtype).dtype),
    )


def _parity(values: Array) -> Array:
    return (jax.lax.population_count(values) & 1).astype(jnp.bool_)


def _affine(symbols: Array, rows: Array, constant: Array) -> Array:
    """Return ``constant ⊕ parity(symbols ∧ rows)`` over the last (word) axis."""
    ones = jnp.sum(jax.lax.population_count(symbols & rows), axis=-1)
    return ((ones & 1) == 1) ^ constant


def outputs(component: StatevectorComponent, state: LaneState) -> Array:
    """Return each lane's output bits, shape ``(L, num_outputs)``."""
    return _affine(
        state.symbols[:, None, :], component.output_row, component.output_const
    )


def _index(component: StatevectorComponent) -> Array:
    return jnp.arange(2**component.width, dtype=jnp.uint32)


def _sign(component: StatevectorComponent, state: LaneState, a: Array) -> Array:
    """Return each lane's sign bit of rotation ``a``."""
    return _affine(state.symbols, component.sign_row[a], component.sign_const[a])


def _apply_pauli(amps: Array, index: Array, x: Array, z: Array) -> Array:
    """Return P·ψ lane by lane, for P = i^|x∧z| X^x Z^z."""
    flipped = index ^ x
    quarter_turns = (jax.lax.population_count(x & z) & 3).astype(jnp.int32)
    i_power = jnp.asarray(_I_POWERS, dtype=amps.dtype)[quarter_turns]
    signs = jnp.where(_parity(flipped & z), -1, 1).astype(amps.dtype)
    return i_power * signs * jnp.take(amps, flipped.astype(jnp.int32), axis=1)


def _rotate_diag(
    component: StatevectorComponent, state: LaneState, a: Array, key: Array
) -> LaneState:
    del key
    odd = _parity(_index(component) & component.z_mask[a])
    flip = _sign(component, state, a)[:, None] ^ odd[None, :]
    factor = component.cos[a] - 1j * component.sin[a] * jnp.where(flip, -1, 1)
    return state._replace(amps=(state.amps * factor).astype(state.amps.dtype))


def _rotate(
    component: StatevectorComponent, state: LaneState, a: Array, key: Array
) -> LaneState:
    del key
    paulied = _apply_pauli(
        state.amps, _index(component), component.x_mask[a], component.z_mask[a]
    )
    negate = jnp.where(_sign(component, state, a), -1, 1)[:, None]
    amps = component.cos[a] * state.amps - 1j * component.sin[a] * negate * paulied
    return state._replace(amps=amps.astype(state.amps.dtype))


def _squared(values: Array) -> Array:
    return jnp.real(values) ** 2 + jnp.imag(values) ** 2


def _set_branch(
    component: StatevectorComponent, state: LaneState, a: Array, outcome: Array
) -> LaneState:
    """Set the measurement's branch symbol to ``outcome``."""
    branch = component.branch[a]
    word = branch // 32
    bit = jnp.uint32(1) << (branch % 32).astype(jnp.uint32)
    flags = jnp.where(outcome, bit, jnp.uint32(0))
    symbols = state.symbols.at[:, word].set(state.symbols[:, word] | flags)
    return state._replace(symbols=symbols)


def _sample_outcome(
    state: LaneState, a: Array, key: Array, weight_0: Array, weight_1: Array
) -> tuple[Array, Array]:
    """Return each lane's outcome and its weight.

    Each outcome's probability is its own weight over the total, so neither is
    computed as one minus the other.
    """
    total = jnp.maximum(weight_0 + weight_1, _TINY)
    key = jax.random.fold_in(key, a)
    outcome = choose_outcome(weight_0 / total, weight_1 / total, key)
    return outcome, jnp.where(outcome, weight_1, weight_0)


def _collapse(
    component: StatevectorComponent,
    state: LaneState,
    a: Array,
    outcome: Array,
    projected: Array,
    norm: Array,
) -> LaneState:
    """Keep the pivot-0 half of the normalized ``projected``; branch on ``outcome``."""
    keep = ((_index(component) >> component.pivot[a].astype(jnp.uint32)) & 1) == 0
    deviation = jnp.maximum(state.max_norm_deviation, jnp.max(jnp.abs(norm - 1)))
    state = state._replace(
        amps=jnp.where(keep[None, :], projected, 0).astype(state.amps.dtype),
        max_norm_deviation=deviation.astype(state.max_norm_deviation.dtype),
    )
    return _set_branch(component, state, a, outcome)


def _measure(
    component: StatevectorComponent, state: LaneState, a: Array, key: Array
) -> LaneState:
    amps = state.amps
    paulied = _apply_pauli(
        amps, _index(component), component.x_mask[a], component.z_mask[a]
    )
    plus, minus = amps + paulied, amps - paulied
    weight_0 = jnp.sum(_squared(plus), axis=1) / 4
    weight_1 = jnp.sum(_squared(minus), axis=1) / 4
    outcome, weight = _sample_outcome(state, a, key, weight_0, weight_1)
    scale = jax.lax.rsqrt(jnp.maximum(2 * weight, _TINY))[:, None]
    projected = jnp.where(outcome[:, None], minus, plus) * scale
    return _collapse(component, state, a, outcome, projected, weight_0 + weight_1)


def _measure_diag(
    component: StatevectorComponent, state: LaneState, a: Array, key: Array
) -> LaneState:
    index = _index(component)
    amps = state.amps
    odd = _parity(index & component.z_mask[a])
    weights = _squared(amps)
    weight_0 = jnp.sum(jnp.where(odd[None, :], 0, weights), axis=1)
    weight_1 = jnp.sum(jnp.where(odd[None, :], weights, 0), axis=1)
    outcome, weight = _sample_outcome(state, a, key, weight_0, weight_1)
    bit = jnp.uint32(1) << component.pivot[a].astype(jnp.uint32)
    partner = jnp.take(amps, (index ^ bit).astype(jnp.int32), axis=1)
    gathered = jnp.where(odd[None, :] == outcome[:, None], amps, partner)
    scale = jax.lax.rsqrt(jnp.maximum(weight, _TINY))[:, None]
    projected = gathered * scale
    return _collapse(component, state, a, outcome, projected, weight_0 + weight_1)


def _measure_dormant(
    component: StatevectorComponent, state: LaneState, a: Array, key: Array
) -> LaneState:
    lanes = state.symbols.shape[0]
    half = jnp.full(lanes, 0.5, dtype=state.max_norm_deviation.dtype)
    outcome = choose_outcome(half, half, jax.random.fold_in(key, a))
    return _set_branch(component, state, a, outcome)


_KERNELS = {
    ROTATE_DIAG: _rotate_diag,
    ROTATE: _rotate,
    MEASURE_DIAG: _measure_diag,
    MEASURE: _measure,
    MEASURE_DORMANT: _measure_dormant,
}


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
    branches = [partial(_KERNELS[kind], component) for kind in range(len(_KERNELS))]

    def step(a: Array, carry: LaneState) -> LaneState:
        return jax.lax.switch(component.kind[a], branches, carry, a, key)

    return jax.lax.fori_loop(start, stop, step, state)


@jax.jit
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
    state = initial_state(component, f_bits)
    state = run(component, state, subkey, 0, component.num_actions)
    return outputs(component, state), key, state.max_norm_deviation
