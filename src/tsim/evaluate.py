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
    # TYPE A: Node Terms (1 + e^(i*alpha))
    # ====================================================================
    rowsum_a = jnp.sum(circuit.a_param_bits * param_vals, axis=1) % 2
    phase_idx_a = (4 * rowsum_a + circuit.a_const_phases) % 8
    term_vals_a = 1 + phase_lut[phase_idx_a]

    summands_a = jax.ops.segment_prod(
        term_vals_a,
        circuit.a_graph_ids,
        num_segments=num_graphs,
        indices_are_sorted=True,
    )

    # ====================================================================
    # TYPE B: Half-Pi Terms (e^(i*beta))
    # ====================================================================
    rowsum_b = jnp.sum(circuit.b_param_bits * param_vals, axis=1) % 2
    # If rowsum is 1, phase is term_type (2 or 6). If 0, phase is 0.
    phase_idx_b = (rowsum_b * circuit.b_term_types) % 8
    term_vals_b = phase_lut[phase_idx_b]

    summands_b = jax.ops.segment_prod(
        term_vals_b,
        circuit.b_graph_ids,
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
        summands_a
        * summands_b
        * summands_c
        * summands_d
        * phase_lut[circuit.phase_indices]
        * root2**circuit.power2
        * circuit.floatfactor
    )

    result = jnp.sum(contributions)
    return result


evaluate_batch = jax.vmap(evaluate, in_axes=(None, 0))
