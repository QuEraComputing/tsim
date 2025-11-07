import jax
import jax.numpy as jnp

from tsim.compile import CompiledCircuit


@jax.jit
def evaluate(circuit: CompiledCircuit, param_vals: jnp.ndarray) -> jnp.ndarray:
    """Evaluate compiled circuit with parameter values.

    Args:
        circuit: Compiled circuit representation
        param_vals: Binary parameter values (error bits + measurement/detector outcomes)

    Returns:
        Complex amplitude for given parameter configuration
    """
    num_graphs = len(circuit.power2)

    # Pre-compute phase lookup table
    phase_lut = jnp.exp(1j * jnp.pi * jnp.arange(8) / 4)

    # ====================================================================
    # TYPE A/B: Node and Half-Pi Terms
    # ====================================================================
    rowsum = jnp.sum(circuit.ab_param_bits * param_vals, axis=1) % 2
    phase_idx = (
        ((4 * rowsum + circuit.ab_const_phases) % 8) * circuit.ab_term_types // 4
    )

    term_vals_ab = phase_lut[phase_idx]
    term_vals_ab = jnp.where(circuit.ab_term_types == 4, term_vals_ab + 1, term_vals_ab)

    summands_ab = jax.ops.segment_prod(
        term_vals_ab,
        circuit.ab_graph_ids,
        num_segments=num_graphs,
        indices_are_sorted=True,
    )

    # ====================================================================
    # TYPE C: Pi-Pair Terms, exp(i*Phi*Psi) where Psi = {0, 1}, Phi = {0, 1}
    # ====================================================================
    rowsum_a = (
        circuit.c_const_bits_a + jnp.sum(circuit.c_param_bits_a * param_vals, axis=1)
    ) % 2
    rowsum_b = (
        circuit.c_const_bits_b + jnp.sum(circuit.c_param_bits_b * param_vals, axis=1)
    ) % 2

    term_vals_c = 1 - 2 * (rowsum_a * rowsum_b).astype(jnp.complex64)

    summands_c = jax.ops.segment_prod(
        term_vals_c, circuit.c_graph_ids, num_segments=num_graphs
    )

    # ====================================================================
    # TYPE D: Phase Pairs
    # ====================================================================
    rowsum_a = jnp.sum(circuit.d_param_bits_a * param_vals, axis=1) % 2
    rowsum_b = jnp.sum(circuit.d_param_bits_b * param_vals, axis=1) % 2

    alpha = (circuit.d_const_alpha + rowsum_a * 4) % 8
    beta = (circuit.d_const_beta + rowsum_b * 4) % 8
    gamma = (alpha + beta) % 8

    term_vals_d = 1.0 + phase_lut[alpha] + phase_lut[beta] - phase_lut[gamma]

    summands_d = jax.ops.segment_prod(
        term_vals_d, circuit.d_graph_ids, num_segments=num_graphs
    )

    # ====================================================================
    # FINAL RESULT
    # ====================================================================
    root2 = jnp.sqrt(2.0)
    contributions = (
        summands_ab
        * summands_c
        * summands_d
        * phase_lut[circuit.phase_indices]
        * root2**circuit.power2
        * circuit.floatfactor
    )

    result = jnp.sum(contributions)
    return result


evaluate_batch = jax.vmap(evaluate, in_axes=(None, 0))
