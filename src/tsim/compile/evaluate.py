"""Evaluation of compiled scalar graphs using exact arithmetic."""

import functools

import jax
import jax.numpy as jnp
from jax import Array

from tsim.compile.compile import CompiledScalarGraphs
from tsim.core.exact_scalar import ExactScalarArray

# Pre-computed exact scalars for phase values, for powers of omega = e^(i*pi/4)
_UNIT_PHASES = jnp.array(
    [
        [1, 0, 0, 0],  # omega^0 = 1
        [0, 1, 0, 0],  # omega^1
        [0, 0, 1, 0],  # omega^2 = i
        [0, 0, 0, -1],  # omega^3
        [-1, 0, 0, 0],  # omega^4 = -1
        [0, -1, 0, 0],  # omega^5
        [0, 0, -1, 0],  # omega^6 = -i
        [0, 0, 0, 1],  # omega^7
    ],
    dtype=jnp.int32,
)

# Lookup table for exact scalars (1 + omega^k)
_ONE_PLUS_PHASES = _UNIT_PHASES.at[:, 0].add(1)

_IDENTITY = jnp.array([1, 0, 0, 0], dtype=jnp.int32)


def _matmul_gf2(a: Array, b: Array) -> Array:
    """Compute binary dot products mod 2 as ``a_GTP x b_BP -> b_BGT``.

    Uses float32 matmul (integer matmul does not have BLAS support on CPU)
    then casts back to uint8.

    Args:
        a: Parameter bit-masks, shape ``(G, T, P)`` — G graphs, T terms, P parameters.
        b: Binary parameter values, shape ``(B, P)`` — B batch elements.

    Returns:
        Binary row-sums mod 2, shape ``(B, G, T)``.

    """
    G, T, _ = a.shape
    if G * T == 0:
        return jnp.zeros((b.shape[0], G, T), dtype=b.dtype)
    return (b.astype(jnp.float32) @ a.astype(jnp.float32).reshape(G * T, -1).T).reshape(
        -1, G, T
    ).astype(jnp.uint8) % 2


@jax.jit
def evaluate(circuit: CompiledScalarGraphs, param_vals: Array) -> Array:
    """Evaluate compiled circuit with batched parameter values.

    Args:
        circuit: Compiled circuit representation
        param_vals: Binary parameter values (error bits + measurement/detector outcomes),
            shape (batch_size, n_params)

    Returns:
        A complex array of shape (batch_size,) containing the amplitudes of the provided
        circuit evaluated with the given binary parameter values.

    """
    # ====================================================================
    # Node phases: product of (1 + e^(i*alpha)) terms.
    # Padded values are masked to multiplicative identity.
    # ====================================================================
    # node.params: (num_graphs, max_terms, n_params), param_vals: (batch_size, n_params)
    node = circuit.node_phases
    rowsum_a = _matmul_gf2(node.params, param_vals)
    phase_idx_a = (4 * rowsum_a + node.phases) % 8

    term_vals_a_exact = _ONE_PLUS_PHASES[phase_idx_a]
    a_mask = jnp.arange(node.phases.shape[1])[None, :] < node.counts[:, None]
    term_vals_a_exact = jnp.where(a_mask[..., None], term_vals_a_exact, _IDENTITY)

    term_vals_a = ExactScalarArray(term_vals_a_exact)
    summands_a = term_vals_a.prod(axis=-1)

    # ====================================================================
    # Half-pi phases: sum of e^(i*beta) terms in the exponent.
    # Padded values are 0, so they don't affect the sum.
    # ====================================================================
    halfpi = circuit.halfpi_phases
    rowsum_b = _matmul_gf2(halfpi.params, param_vals)
    phase_idx_b = (rowsum_b * halfpi.coeffs) % 8

    sum_phases_b = jnp.sum(phase_idx_b, axis=-1) % 8

    summands_b_exact = _UNIT_PHASES[sum_phases_b]
    summands_b = ExactScalarArray(summands_b_exact)

    # ====================================================================
    # Pi products: (-1)^(Psi*Phi) terms.
    # ====================================================================
    pi = circuit.pi_products
    rowsum_a_c = (pi.psi_const + _matmul_gf2(pi.psi_params, param_vals)) % 2
    rowsum_b_c = (pi.phi_const + _matmul_gf2(pi.phi_params, param_vals)) % 2

    exponent_c = (rowsum_a_c * rowsum_b_c) % 2
    sum_exponents_c = jnp.sum(exponent_c, axis=-1) % 2

    summands_c_exact = (1 - 2 * sum_exponents_c)[..., None] * jnp.array(
        [1, 0, 0, 0], dtype=jnp.int32
    )
    summands_c = ExactScalarArray(summands_c_exact)

    # ====================================================================
    # Phase pairs: product of (1 + e^a + e^b - e^(a+b)) terms.
    # Padded values are masked to multiplicative identity.
    # ====================================================================
    pair = circuit.phase_pairs
    rowsum_a_d = _matmul_gf2(pair.alpha_params, param_vals)
    rowsum_b_d = _matmul_gf2(pair.beta_params, param_vals)

    alpha = (pair.alpha + rowsum_a_d * 4) % 8
    beta = (pair.beta + rowsum_b_d * 4) % 8
    gamma = (alpha + beta) % 8

    term_vals_d_exact = (
        _IDENTITY + _UNIT_PHASES[alpha] + _UNIT_PHASES[beta] - _UNIT_PHASES[gamma]
    )
    d_mask = jnp.arange(pair.alpha.shape[1])[None, :] < pair.counts[:, None]
    term_vals_d_exact = jnp.where(d_mask[..., None], term_vals_d_exact, _IDENTITY)

    term_vals_d = ExactScalarArray(term_vals_d_exact)
    summands_d = term_vals_d.prod(axis=-1)

    # ====================================================================
    # FINAL COMBINATION
    # ====================================================================
    static_phases = ExactScalarArray(_UNIT_PHASES[circuit.phase_indices])
    float_factor = ExactScalarArray(circuit.floatfactor)

    total_summands = functools.reduce(
        lambda a, b: a * b,
        [summands_a, summands_b, summands_c, summands_d, static_phases, float_factor],
    )

    if not circuit.has_approximate_floatfactors:
        total_summands = ExactScalarArray(
            total_summands.coeffs, total_summands.power + circuit.power2
        )
        return total_summands.sum().to_complex()
    else:
        return jnp.sum(
            total_summands.to_complex()
            * circuit.approximate_floatfactors
            * 2.0**circuit.power2,
            axis=-1,
        )
