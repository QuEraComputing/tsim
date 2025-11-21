import jax
import jax.numpy as jnp

from tsim.compile import CompiledCircuit
from tsim.exact_scalar import DyadicArray


@jax.jit
def evaluate(circuit: CompiledCircuit, param_vals: jnp.ndarray) -> DyadicArray:
    """Evaluate compiled circuit with parameter values.

    Args:
        circuit: Compiled circuit representation
        param_vals: Binary parameter values (error bits + measurement/detector outcomes)

    Returns:
        Complex amplitude for given parameter configuration
    """
    num_graphs = len(circuit.power2)

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
    # ====================================================================
    rowsum_a = jnp.sum(circuit.a_param_bits * param_vals, axis=1) % 2
    phase_idx_a = (4 * rowsum_a + circuit.a_const_phases) % 8

    term_vals_a_exact = one_plus_phases_exact[phase_idx_a]

    term_vals_a = DyadicArray(term_vals_a_exact)
    summands_a = term_vals_a.segment_prod(
        circuit.a_graph_ids,
        num_segments=num_graphs,
        indices_are_sorted=True,
    )

    # ====================================================================
    # TYPE B: Half-Pi Terms (e^(i*beta))
    # ====================================================================
    # For Type B (monomials), we can sum indices modulo 8 instead of multiplying scalars

    rowsum_b = jnp.sum(circuit.b_param_bits * param_vals, axis=1) % 2
    phase_idx_b = (rowsum_b * circuit.b_term_types) % 8

    sum_phases_b = (
        jax.ops.segment_sum(
            phase_idx_b,
            circuit.b_graph_ids,
            num_segments=num_graphs,
            indices_are_sorted=True,
        )
        % 8
    )

    # Convert final summed phase to ExactScalar
    summands_b_exact = unit_phases_exact[sum_phases_b]
    summands_b = DyadicArray(summands_b_exact)

    # ====================================================================
    # TYPE C: Pi-Pair Terms, (-1)^(Psi*Phi)
    # ====================================================================
    # These are +/- 1.

    rowsum_a = (
        circuit.c_const_bits_a + jnp.sum(circuit.c_param_bits_a * param_vals, axis=1)
    ) % 2
    rowsum_b = (
        circuit.c_const_bits_b + jnp.sum(circuit.c_param_bits_b * param_vals, axis=1)
    ) % 2

    exponent_c = (rowsum_a * rowsum_b) % 2

    sum_exponents_c = (
        jax.ops.segment_sum(
            exponent_c,
            circuit.c_graph_ids,
            num_segments=num_graphs,
            indices_are_sorted=True,
        )
        % 2
    )

    # Map 0 -> 1, 1 -> -1
    # 1  = [1, 0, 0, 0]
    # -1 = [-1, 0, 0, 0]
    # Vectorized: set 'a' component to 1 - 2*exponent
    summands_c_exact = jnp.zeros((num_graphs, 4), dtype=jnp.int32)
    summands_c_exact = summands_c_exact.at[:, 0].set(1 - 2 * sum_exponents_c)
    summands_c = DyadicArray(summands_c_exact)

    # ====================================================================
    # TYPE D: Phase Pairs (1 + e^a + e^b - e^g)
    # ====================================================================
    rowsum_a = jnp.sum(circuit.d_param_bits_a * param_vals, axis=1) % 2
    rowsum_b = jnp.sum(circuit.d_param_bits_b * param_vals, axis=1) % 2

    alpha = (circuit.d_const_alpha + rowsum_a * 4) % 8
    beta = (circuit.d_const_beta + rowsum_b * 4) % 8
    gamma = (alpha + beta) % 8

    # 1 + e^a + e^b - e^g
    term_vals_d_exact = (
        jnp.array([1, 0, 0, 0], dtype=jnp.int32)
        + unit_phases_exact[alpha]
        + unit_phases_exact[beta]
        - unit_phases_exact[gamma]
    )

    term_vals_d = DyadicArray(term_vals_d_exact)
    summands_d = term_vals_d.segment_prod(
        circuit.d_graph_ids,
        num_segments=num_graphs,
        indices_are_sorted=True,
    )

    # ====================================================================
    # FINAL COMBINATION
    # ====================================================================

    static_phases = DyadicArray(unit_phases_exact[circuit.phase_indices])
    float_factor = DyadicArray(circuit.floatfactor)

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

    # Add initial power2 from circuit compilation
    total_summands.power = total_summands.power + circuit.power2

    total_summands = total_summands.reduce()
    return total_summands.sum()


evaluate_batch = jax.vmap(evaluate, in_axes=(None, 0))
