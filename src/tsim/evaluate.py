import functools
from typing import Literal, overload

import jax
import jax.numpy as jnp
from jax import Array

from tsim.compile import CompiledScalarGraphs
from tsim.exact_scalar import ExactScalarArray


@overload
def evaluate(
    circuit: CompiledScalarGraphs,
    param_vals: Array,
    has_approximate_floatfactor: Literal[False],
) -> ExactScalarArray: ...


@overload
def evaluate(
    circuit: CompiledScalarGraphs,
    param_vals: Array,
    has_approximate_floatfactor: Literal[True],
) -> Array: ...


@overload
def evaluate(
    circuit: CompiledScalarGraphs,
    param_vals: Array,
    has_approximate_floatfactor: bool,
) -> ExactScalarArray | Array: ...


@functools.partial(jax.jit, static_argnums=(2,))
def evaluate(
    circuit: CompiledScalarGraphs, param_vals: Array, has_approximate_floatfactor: bool
) -> ExactScalarArray | Array:
    """Evaluate compiled circuit with parameter values.

    Args:
        circuit: Compiled circuit representation
        param_vals: Binary parameter values (error bits + measurement/detector outcomes),
            shape (n_params,)
        has_approximate_floatfactor: Whether the circuit has approximate float factors.
            Determines the return type and evaluation strategy.

    Returns:
        ExactScalarArray if has_approximate_floatfactor is False, otherwise a complex Array
        representing the amplitude for the given parameter configuration.
    """
    num_graphs = circuit.power2.shape[0]

    # Pre-compute exact scalars for phase values, for powers of omega = e^(i*pi/4)
    unit_phases_exact = jnp.array(
        [
            [1, 0, 0, 0],  # 0: 1
            [0, 1, 0, 0],  # 1: e^i
            [0, 0, 1, 0],  # 2: i
            [0, 0, 0, -1],  # 3: e^3pi/4 = -e^-pi/4
            [-1, 0, 0, 0],  # 4: -1
            [0, -1, 0, 0],  # 5: e^5pi/4 =-e^pi/4
            [0, 0, -1, 0],  # 6: -i
            [0, 0, 0, 1],  # 7: -i*e^pi/4 = e^-pi/4
        ],
        dtype=jnp.int32,
    )

    # Lookup table for exact scalars (1 + omega^k)
    one_plus_phases_exact = unit_phases_exact.at[:, 0].add(1)

    # ====================================================================
    # TYPE A: Node Terms (1 + e^(i*alpha))
    # Shape: (num_graphs, max_a) -> (num_graphs, max_a, 4) -> prod -> (num_graphs, 4)
    # Padded values are masked to multiplicative identity.
    # ====================================================================
    # a_param_bits: (num_graphs, max_a, n_params), param_vals: (n_params,)
    # Broadcast: (num_graphs, max_a, n_params) * (n_params,) -> sum over last axis
    rowsum_a = jnp.sum(circuit.a_param_bits * param_vals, axis=-1) % 2
    phase_idx_a = (4 * rowsum_a + circuit.a_const_phases) % 8

    term_vals_a_exact = one_plus_phases_exact[phase_idx_a]  # (num_graphs, max_a, 4)
    a_mask = (
        jnp.arange(circuit.a_const_phases.shape[1])[None, :]
        < circuit.a_num_terms[:, None]
    )
    term_vals_a_exact = jnp.where(
        a_mask[..., None], term_vals_a_exact, jnp.array([1, 0, 0, 0], dtype=jnp.int32)
    )

    term_vals_a = ExactScalarArray(term_vals_a_exact)
    summands_a = term_vals_a.prod(axis=1)  # (num_graphs, 4)

    # ====================================================================
    # TYPE B: Half-Pi Terms (e^(i*beta))
    # For Type B (monomials), we can sum indices modulo 8 instead of multiplying scalars
    # Padded values are 0, so they don't affect the sum.
    # ====================================================================
    rowsum_b = jnp.sum(circuit.b_param_bits * param_vals, axis=-1) % 2
    phase_idx_b = (rowsum_b * circuit.b_term_types) % 8  # (num_graphs, max_b)

    sum_phases_b = jnp.sum(phase_idx_b, axis=1) % 8  # (num_graphs,)

    # Convert final summed phase to ExactScalar
    summands_b_exact = unit_phases_exact[sum_phases_b]  # (num_graphs, 4)
    summands_b = ExactScalarArray(summands_b_exact)

    # ====================================================================
    # TYPE C: Pi-Pair Terms, (-1)^(Psi*Phi)
    # These are +/- 1. Padded values contribute 0 to the exponent sum.
    # ====================================================================
    rowsum_a_c = (
        circuit.c_const_bits_a + jnp.sum(circuit.c_param_bits_a * param_vals, axis=-1)
    ) % 2
    rowsum_b_c = (
        circuit.c_const_bits_b + jnp.sum(circuit.c_param_bits_b * param_vals, axis=-1)
    ) % 2

    exponent_c = (rowsum_a_c * rowsum_b_c) % 2  # (num_graphs, max_c)

    sum_exponents_c = jnp.sum(exponent_c, axis=1) % 2  # (num_graphs,)

    # Map 0 -> 1, 1 -> -1
    summands_c_exact = jnp.zeros((num_graphs, 4), dtype=jnp.int32)
    summands_c_exact = summands_c_exact.at[:, 0].set(1 - 2 * sum_exponents_c)
    summands_c = ExactScalarArray(summands_c_exact)

    # ====================================================================
    # TYPE D: Phase Pairs (1 + e^a + e^b - e^g)
    # Padded values are masked to multiplicative identity.
    # ====================================================================
    rowsum_a_d = jnp.sum(circuit.d_param_bits_a * param_vals, axis=-1) % 2
    rowsum_b_d = jnp.sum(circuit.d_param_bits_b * param_vals, axis=-1) % 2

    alpha = (circuit.d_const_alpha + rowsum_a_d * 4) % 8
    beta = (circuit.d_const_beta + rowsum_b_d * 4) % 8
    gamma = (alpha + beta) % 8

    # 1 + e^a + e^b - e^g, shape: (num_graphs, max_d, 4)
    term_vals_d_exact = (
        jnp.array([1, 0, 0, 0], dtype=jnp.int32)
        + unit_phases_exact[alpha]
        + unit_phases_exact[beta]
        - unit_phases_exact[gamma]
    )
    d_mask = (
        jnp.arange(circuit.d_const_alpha.shape[1])[None, :]
        < circuit.d_num_terms[:, None]
    )
    term_vals_d_exact = jnp.where(
        d_mask[..., None], term_vals_d_exact, jnp.array([1, 0, 0, 0], dtype=jnp.int32)
    )

    term_vals_d = ExactScalarArray(term_vals_d_exact)
    summands_d = term_vals_d.prod(axis=1)  # (num_graphs, 4)

    # ====================================================================
    # FINAL COMBINATION
    # ====================================================================

    static_phases = ExactScalarArray(unit_phases_exact[circuit.phase_indices])
    float_factor = ExactScalarArray(circuit.floatfactor)

    def mul_all(terms):
        res = terms[0]
        for t in terms[1:]:
            res = res * t
        return res

    total_summands = mul_all(
        [
            summands_a,
            summands_b,
            summands_c,
            summands_d,
            static_phases,
            float_factor,
        ]
    )

    if not has_approximate_floatfactor:
        # Add initial power2 from circuit compilation
        total_summands = ExactScalarArray(
            total_summands.coeffs, total_summands.power + circuit.power2
        )
        total_summands = total_summands.reduce()
        return total_summands.sum()
    else:
        return jnp.sum(
            total_summands.to_complex()
            * circuit.approximate_floatfactors
            * 2.0**circuit.power2,
            axis=-1,
        )


_evaluate_batch = jax.vmap(evaluate, in_axes=(None, 0, None))


def evaluate_batch(circuit: CompiledScalarGraphs, param_vals: Array) -> Array:
    """Evaluate compiled circuit with batched parameters, returning JAX array."""
    if circuit.has_approximate_floatfactors:
        return _evaluate_batch(circuit, param_vals, True)
    return _evaluate_batch(circuit, param_vals, False).to_complex()
