import jax
import jax.numpy as jnp

from tsim.compile import CompiledCircuit
from tsim.exact_scalar import scalar_mul, segment_scalar_prod


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

    # Pre-compute exact scalars for phase values 0..7
    # omega^k for k=0..7 in Z[omega] representation [a, b, c, d]
    # 1         -> [1, 0, 0, 0]
    # omega     -> [0, 1, 0, 0]
    # i         -> [0, 0, 1, 0]
    # i*omega   -> [0, 0, 0, -1]
    # Basis is: 1, omega, i, omega^-1 (which is -i*omega)
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

    # Lookup table for (1 + omega^k) [0..7] -> ExactScalar
    # Just add [1, 0, 0, 0] to unit_phases_exact
    one_plus_phases_exact = unit_phases_exact.at[:, 0].add(1)

    # ====================================================================
    # TYPE A: Node Terms (1 + e^(i*alpha))
    # ====================================================================
    rowsum_a = jnp.sum(circuit.a_param_bits * param_vals, axis=1) % 2
    phase_idx_a = (4 * rowsum_a + circuit.a_const_phases) % 8

    term_vals_a_exact = one_plus_phases_exact[phase_idx_a]

    summands_a_exact = segment_scalar_prod(
        term_vals_a_exact,
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

    # ====================================================================
    # TYPE C: Pi-Pair Terms, (-1)^(Psi*Phi)
    # ====================================================================
    # These are +/- 1.
    # (-1)^x * (-1)^y = (-1)^(x+y)
    # So we can sum the exponents modulo 2.

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
    c_coeffs = jnp.zeros((num_graphs, 4), dtype=jnp.int32)
    c_coeffs = c_coeffs.at[:, 0].set(1 - 2 * sum_exponents_c)
    summands_c_exact = c_coeffs

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

    summands_d_exact = segment_scalar_prod(
        term_vals_d_exact,
        circuit.d_graph_ids,
        num_segments=num_graphs,
        indices_are_sorted=True,
    )

    # ====================================================================
    # FINAL COMBINATION
    # ====================================================================

    static_phases_exact = unit_phases_exact[circuit.phase_indices]

    def mul_all(terms):
        res = terms[0]
        for t in terms[1:]:
            res = scalar_mul(res, t)
        return res

    total_exact = mul_all(
        [
            summands_a_exact,
            summands_b_exact,
            summands_c_exact,
            summands_d_exact,
            static_phases_exact,
            circuit.floatfactor,
        ]
    )

    power2 = circuit.power2
    for i in range(10):
        reducible = jnp.all(total_exact % 2 == 0, axis=-1)
        total_exact = jnp.where(reducible[..., None], total_exact // 2, total_exact)
        power2 = jnp.where(reducible, power2 + 1, power2)

    min_power2 = jnp.min(power2, keepdims=True)
    pow = (power2 - min_power2)[:, None]
    total_exact2 = total_exact * 2**pow

    exact_sum = jnp.sum(total_exact2, axis=-2)

    # ====================================================================
    # FINAL RESULT (Conversion to Complex)
    # ====================================================================

    # scale_factor = jnp.pow(2.0, min_power2.astype(jnp.float64))
    # complex_sum = scalar_to_complex(exact_sum)
    # result = complex_sum * scale_factor

    return jnp.hstack((-min_power2, exact_sum))


evaluate_batch = jax.vmap(evaluate, in_axes=(None, 0))


# def evaluate_batch(circuit: CompiledCircuit, param_vals: jnp.ndarray) -> jnp.ndarray:
#     res = []
#     for p in param_vals:
#         res.append(evaluate(circuit, p))
#     return jnp.array(res)
