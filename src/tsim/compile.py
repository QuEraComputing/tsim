from collections import defaultdict
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from pyzx.graph.base import BaseGraph
from pyzx.graph.scalar import DyadicNumber


class CompiledScalarGraphs(eqx.Module):
    """JAX-compatible compiled scalar graphs representation.

    All fields are static-shaped JAX arrays.

    Term arrays are 2D with shape (num_graphs, max_terms_per_graph) and padded
    with identity values (1 for products, 0 for sums) for graphs with fewer terms.
    """

    # Metadata
    num_graphs: int
    n_params: int

    # Type A: Node Terms (1 + e^(i*alpha))
    # Padded with const_phase=0 which gives (1+e^0)=2; compensated in power2
    a_const_phases: Array  # shape: (num_graphs, max_a), dtype: uint8, values: 0-7
    a_param_bits: Array  # shape: (num_graphs, max_a, n_params), dtype: uint8

    # Type B: Half-Pi Terms (e^(i*beta))
    b_term_types: Array  # shape: (num_graphs, max_b), dtype: uint8, values: {0,2,4,6}
    b_param_bits: Array  # shape: (num_graphs, max_b, n_params), dtype: uint8

    # Type C: Pi-Pair Terms (e^(i*Psi*Phi))
    c_const_bits_a: Array  # shape: (num_graphs, max_c), dtype: uint8, values: {0, 1}
    c_param_bits_a: Array  # shape: (num_graphs, max_c, n_params), dtype: uint8
    c_const_bits_b: Array  # shape: (num_graphs, max_c), dtype: uint8, values: {0, 1}
    c_param_bits_b: Array  # shape: (num_graphs, max_c, n_params), dtype: uint8

    # Type D: Phase Pairs (1 + e^a + e^b - e^(a+b))
    # Padded with alpha=0, beta=0 which gives 1+1+1-1=2; compensated in power2
    d_const_alpha: Array  # shape: (num_graphs, max_d), dtype: uint8, values: 0-7
    d_const_beta: Array  # shape: (num_graphs, max_d), dtype: uint8, values: 0-7
    d_param_bits_a: Array  # shape: (num_graphs, max_d, n_params), dtype: uint8
    d_param_bits_b: Array  # shape: (num_graphs, max_d, n_params), dtype: uint8

    # Static per-graph data
    phase_indices: Array  # shape: (num_graphs,), dtype: uint8 (values 0-7)
    has_approximate_floatfactors: bool = eqx.field(static=True)
    # TODO: use complex128
    approximate_floatfactors: Array  # shape: (num_graphs,), dtype: complex64
    power2: Array  # shape: (num_graphs,), dtype: int32
    floatfactor: Array  # shape: (num_graphs, 4), dtype: int32


def compile_scalar_graphs(
    g_list: list[BaseGraph], params: list[str]
) -> CompiledScalarGraphs:
    """Compile ZX-graph list into JAX-compatible structure for fast evaluation.

    Args:
        g_list: List of ZX-graphs to compile (must be scalar graphs with no vertices)
        params: List of parameter names used by this circuit. Each parameter will correspond to columns in
            the jax.Arrays of the compiled circuit.

    Returns:
        CompiledScalarGraphs with all data in static-shaped JAX arrays
    """
    for i, g in enumerate(g_list):
        assert (
            len(list(g.vertices())) == 0
        ), f"Only scalar graphs can be compiled but graph {i} has {len(list(g.vertices()))} vertices"

    g_list = [g for g in g_list if not g.scalar.is_zero]

    n_params = len(params)
    num_graphs = len(g_list)
    char_to_idx = {char: i for i, char in enumerate(params)}

    # ========================================================================
    # Type A compilation (phase-node)
    # Terms of the form exp(i * (α + parity(params)) * pi).
    # Collect per-graph lists, then pad to 2D arrays.
    # ========================================================================
    a_terms_per_graph: list[list[tuple[int, list[int]]]] = [
        [] for _ in range(num_graphs)
    ]

    for i in range(num_graphs):
        g_i = g_list[i]
        for term in range(len(g_i.scalar.phasenodevars)):
            bitstr = [0] * n_params
            for v in g_i.scalar.phasenodevars[term]:
                bitstr[char_to_idx[v]] = 1
            assert g_i.scalar.phasenodes[term].denominator in [1, 2, 4]
            const_term = int(g_i.scalar.phasenodes[term] * 4)  # type: ignore[arg-type]
            a_terms_per_graph[i].append((const_term, bitstr))

    max_a = max((len(terms) for terms in a_terms_per_graph), default=0)
    max_a = max(max_a, 1)

    # Pad with const_phase=0, param_bits=0 -> (1+e^0)=2; compensated in power2.
    a_const_phases = np.zeros((num_graphs, max_a), dtype=np.uint8)
    a_param_bits = np.zeros((num_graphs, max_a, n_params), dtype=np.uint8)

    for i, terms in enumerate(a_terms_per_graph):
        for j, (const_phase, param_bit) in enumerate(terms):
            a_const_phases[i, j] = const_phase
            a_param_bits[i, j] = param_bit

    # ========================================================================
    # Type B compilation (half-π)
    # Phase terms of the form exp(1j * π * j * parity(params) / 2) where j ∈ {1, 3}.
    # ========================================================================
    b_terms_per_graph: list[list[tuple[int, list[int]]]] = [
        [] for _ in range(num_graphs)
    ]

    for i in range(num_graphs):
        g_i = g_list[i]
        assert set(g_i.scalar.phasevars_halfpi.keys()) <= {1, 3}

        # Accumulate j values per bitstring for this graph
        bitstr_to_j: dict[tuple[int, ...], int] = defaultdict(int)

        for j in [1, 3]:
            if j not in g_i.scalar.phasevars_halfpi:
                continue
            for term in range(len(g_i.scalar.phasevars_halfpi[j])):
                bitstr = [0] * n_params
                for v in g_i.scalar.phasevars_halfpi[j][term]:
                    bitstr[char_to_idx[v]] = 1
                bitstr_key = tuple(bitstr)
                bitstr_to_j[bitstr_key] = (bitstr_to_j[bitstr_key] + j) % 4

        for bitstr_key, combined_j in bitstr_to_j.items():
            if combined_j == 0:
                continue
            b_terms_per_graph[i].append((combined_j * 2, list(bitstr_key)))

    max_b = max((len(terms) for terms in b_terms_per_graph), default=0)
    max_b = max(max_b, 1)

    # Pad with 0 (additive identity for phase sums)
    b_term_types = np.zeros((num_graphs, max_b), dtype=np.uint8)
    b_param_bits = np.zeros((num_graphs, max_b, n_params), dtype=np.uint8)

    for i, terms in enumerate(b_terms_per_graph):
        for j, (term_type, param_bit) in enumerate(terms):
            b_term_types[i, j] = term_type
            b_param_bits[i, j] = param_bit

    # ========================================================================
    # Type C compilation (π-pair)
    # Terms of the form (-1)^(ψ * φ) where ψ and φ are parities.
    # ========================================================================
    c_terms_per_graph: list[list[tuple[int, list[int], int, list[int]]]] = [
        [] for _ in range(num_graphs)
    ]

    for i in range(num_graphs):
        graph = g_list[i]
        for p_set in graph.scalar.phasevars_pi_pair:
            const_bit_a = 1 if "1" in p_set[0] else 0
            param_bits_a = [0] * n_params
            for p in p_set[0]:
                if p != "1":
                    param_bits_a[char_to_idx[p]] = 1

            const_bit_b = 1 if "1" in p_set[1] else 0
            param_bits_b = [0] * n_params
            for p in p_set[1]:
                if p != "1":
                    param_bits_b[char_to_idx[p]] = 1

            c_terms_per_graph[i].append(
                (const_bit_a, param_bits_a, const_bit_b, param_bits_b)
            )

    max_c = max((len(terms) for terms in c_terms_per_graph), default=0)
    max_c = max(max_c, 1)

    # Pad with 0 (additive identity for exponent sums)
    c_const_bits_a = np.zeros((num_graphs, max_c), dtype=np.uint8)
    c_param_bits_a = np.zeros((num_graphs, max_c, n_params), dtype=np.uint8)
    c_const_bits_b = np.zeros((num_graphs, max_c), dtype=np.uint8)
    c_param_bits_b = np.zeros((num_graphs, max_c, n_params), dtype=np.uint8)

    for i, terms in enumerate(c_terms_per_graph):
        for j, (cba, pba, cbb, pbb) in enumerate(terms):
            c_const_bits_a[i, j] = cba
            c_param_bits_a[i, j] = pba
            c_const_bits_b[i, j] = cbb
            c_param_bits_b[i, j] = pbb

    # ========================================================================
    # Type D compilation (phase-pair)
    # Terms of the form 1 + e^(i*alpha) + e^(i*beta) - e^(i*(alpha+beta))
    # ========================================================================
    d_terms_per_graph: list[list[tuple[int, int, list[int], list[int]]]] = [
        [] for _ in range(num_graphs)
    ]

    for i in range(num_graphs):
        graph = g_list[i]
        for pp in range(len(graph.scalar.phasepairs)):
            param_bits_a = [0] * n_params
            for v in graph.scalar.phasepairs[pp].paramsA:
                param_bits_a[char_to_idx[v]] = 1

            param_bits_b = [0] * n_params
            for v in graph.scalar.phasepairs[pp].paramsB:
                param_bits_b[char_to_idx[v]] = 1

            const_alpha = int(graph.scalar.phasepairs[pp].alpha)
            const_beta = int(graph.scalar.phasepairs[pp].beta)

            d_terms_per_graph[i].append(
                (const_alpha, const_beta, param_bits_a, param_bits_b)
            )

    max_d = max((len(terms) for terms in d_terms_per_graph), default=0)
    max_d = max(max_d, 1)

    # Pad with alpha=0, beta=0 -> 1+1+1-1=2; compensated in power2.
    d_const_alpha = np.zeros((num_graphs, max_d), dtype=np.uint8)
    d_const_beta = np.zeros((num_graphs, max_d), dtype=np.uint8)
    d_param_bits_a = np.zeros((num_graphs, max_d, n_params), dtype=np.uint8)
    d_param_bits_b = np.zeros((num_graphs, max_d, n_params), dtype=np.uint8)

    for i, terms in enumerate(d_terms_per_graph):
        for j, (ca, cb, pba, pbb) in enumerate(terms):
            d_const_alpha[i, j] = ca
            d_const_beta[i, j] = cb
            d_param_bits_a[i, j] = pba
            d_param_bits_b[i, j] = pbb

    # ========================================================================
    # Static data
    # ========================================================================
    for g in g_list:
        if g.scalar.phase.denominator not in [1, 2, 4]:
            g.scalar.approximate_floatfactor *= np.exp(1j * g.scalar.phase * np.pi)
            g.scalar.phase = Fraction(0, 1)

    has_approximate_floatfactors = any(
        g.scalar.approximate_floatfactor != 1.0 for g in g_list
    )
    approximate_floatfactors = jnp.array(
        [g.scalar.approximate_floatfactor for g in g_list], dtype=jnp.complex64
    )

    phase_indices = jnp.array(
        [int(float(g.scalar.phase) * 4) for g in g_list], dtype=jnp.uint8
    )

    exact_floatfactor = []
    power2 = []

    for i, g in enumerate(g_list):
        dn = g.scalar.floatfactor.copy()

        p_sqrt2 = g.scalar.power2

        if p_sqrt2 % 2 != 0:
            p_sqrt2 -= 1
            dn *= DyadicNumber(k=0, a=0, b=1, c=0, d=1)

        assert p_sqrt2 % 2 == 0
        p_sqrt2 -= 2 * dn.k
        dn.k = 0

        # Padding compensation: each padded A/D term contributes factor of 2.
        # power2 is in powers of 2, so subtract 1 per padded term.
        n_padded_a = max_a - len(a_terms_per_graph[i])
        n_padded_d = max_d - len(d_terms_per_graph[i])
        padding_compensation = n_padded_a + n_padded_d

        power2.append(p_sqrt2 // 2 - padding_compensation)
        exact_floatfactor.append([dn.a, dn.b, dn.c, dn.d])

    return CompiledScalarGraphs(
        num_graphs=num_graphs,
        n_params=n_params,
        a_const_phases=jnp.array(a_const_phases),
        a_param_bits=jnp.array(a_param_bits),
        b_term_types=jnp.array(b_term_types),
        b_param_bits=jnp.array(b_param_bits),
        c_const_bits_a=jnp.array(c_const_bits_a),
        c_param_bits_a=jnp.array(c_param_bits_a),
        c_const_bits_b=jnp.array(c_const_bits_b),
        c_param_bits_b=jnp.array(c_param_bits_b),
        d_const_alpha=jnp.array(d_const_alpha),
        d_const_beta=jnp.array(d_const_beta),
        d_param_bits_a=jnp.array(d_param_bits_a),
        d_param_bits_b=jnp.array(d_param_bits_b),
        phase_indices=phase_indices,
        has_approximate_floatfactors=has_approximate_floatfactors,
        approximate_floatfactors=approximate_floatfactors,
        power2=jnp.array(power2, dtype=jnp.int32),
        floatfactor=jnp.array(exact_floatfactor, dtype=jnp.int32),
    )
