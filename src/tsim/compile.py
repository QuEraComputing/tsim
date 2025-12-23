from collections import defaultdict
from fractions import Fraction
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from pyzx.graph.base import BaseGraph
from pyzx.graph.scalar import DyadicNumber


class CompiledCircuit(NamedTuple):
    """JAX-compatible compiled circuit representation.

    All fields are static-shaped NumPy arrays, making them directly
    convertible to JAX arrays for GPU execution and JIT compilation.

    Constants and parameter-dependent data are separated for cleaner evaluation.
    """

    # Metadata
    num_graphs: int
    n_params: int

    # Type A: Node Terms (1 + e^(i*alpha))
    a_const_phases: jnp.ndarray  # shape: (n_a_terms,), dtype: uint8, values: 0-7
    a_param_bits: jnp.ndarray  # shape: (n_a_terms, n_params), dtype: uint8
    a_graph_ids: jnp.ndarray  # shape: (n_a_terms,), dtype: int32

    # Type B: Half-Pi Terms (e^(i*beta))
    b_term_types: jnp.ndarray  # shape: (n_b_terms,), dtype: uint8, values: {2, 4, 6}
    b_param_bits: jnp.ndarray  # shape: (n_b_terms, n_params), dtype: uint8
    b_graph_ids: jnp.ndarray  # shape: (n_b_terms,), dtype: int32

    # Type C: Pi-Pair Terms (e^(i*Psi*Phi))
    c_const_bits_a: jnp.ndarray  # shape: (n_c_terms,), dtype: uint8, values: {0, 1}
    c_param_bits_a: jnp.ndarray  # shape: (n_c_terms, n_params), dtype: uint8
    c_const_bits_b: jnp.ndarray  # shape: (n_c_terms,), dtype: uint8, values: {0, 1}
    c_param_bits_b: jnp.ndarray  # shape: (n_c_terms, n_params), dtype: uint8
    c_graph_ids: jnp.ndarray  # shape: (n_c_terms,), dtype: int32

    # Type D: Phase Pairs (1 + e^a + e^b - e^(a+b))
    d_const_alpha: jnp.ndarray  # shape: (n_d_terms,), dtype: uint8, values: 0-7
    d_const_beta: jnp.ndarray  # shape: (n_d_terms,), dtype: uint8, values: 0-7
    d_param_bits_a: jnp.ndarray  # shape: (n_d_terms, n_params), dtype: uint8
    d_param_bits_b: jnp.ndarray  # shape: (n_d_terms, n_params), dtype: uint8
    d_graph_ids: jnp.ndarray  # shape: (n_d_terms,), dtype: int32

    # Static per-graph data
    phase_indices: jnp.ndarray  # shape: (num_graphs,), dtype: uint8 (values 0-7)
    has_approximate_floatfactors: bool  # shape: (1,), dtype: bool
    # TODO: use complex128
    approximate_floatfactors: jnp.ndarray  # shape: (num_graphs,), dtype: complex64
    power2: jnp.ndarray  # shape: (num_graphs,), dtype: int32
    floatfactor: jnp.ndarray  # shape: (num_graphs, 4), dtype: int32


def compile_circuit(g_list: list[BaseGraph], params: list[str]) -> CompiledCircuit:
    """Compile ZX-graph list into JAX-compatible structure for fast evaluation.

    Args:
        g_list: List of ZX-graphs to compile (must be scalar graphs with no vertices)
        params: List of parameter names used by this circuit. Each parameter will correspond to columns in
            the jax.Arrays of the compiled circuit.

    Returns:
        CompiledCircuit with all data in static-shaped arrays
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
    # ========================================================================
    a_const_phases_list = []
    a_param_bits_list = []
    g_coord_a = []

    for i in range(num_graphs):
        g_i = g_list[i]
        for term in range(len(g_i.scalar.phasenodevars)):
            bitstr = [0] * n_params
            for v in g_i.scalar.phasenodevars[term]:
                bitstr[char_to_idx[v]] = 1
            assert g_i.scalar.phasenodes[term].denominator in [1, 2, 4]
            const_term = int(g_i.scalar.phasenodes[term] * 4)  # type: ignore[arg-type]

            g_coord_a.append(i)
            a_const_phases_list.append(const_term)
            a_param_bits_list.append(bitstr)

    a_const_phases = jnp.array(a_const_phases_list, dtype=jnp.uint8)
    a_param_bits = jnp.array(a_param_bits_list, dtype=jnp.uint8).reshape(
        len(a_param_bits_list), n_params
    )
    a_graph_ids = jnp.array(g_coord_a, dtype=jnp.int32)

    # ========================================================================
    # Type B compilation (half-π)
    # Phase terms of the form exp(1j * π * j * parity(params) / 2) where j ∈ {1, 3}.
    # ========================================================================
    b_term_types_list = []
    b_param_bits_list = []
    g_coord_b = []

    for i in range(num_graphs):
        g_i = g_list[i]
        assert set(g_i.scalar.phasevars_halfpi.keys()) <= {1, 3}

        # Accumulate j values per bitstring for this graph
        # Key: tuple(bitstr), Value: accumulated j mod 4
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
                # Discard terms that cancelled out
                continue
            g_coord_b.append(i)
            b_term_types_list.append(combined_j * 2)
            b_param_bits_list.append(list(bitstr_key))

    b_term_types = jnp.array(b_term_types_list, dtype=jnp.uint8)
    b_param_bits = jnp.array(b_param_bits_list, dtype=jnp.uint8).reshape(
        len(b_param_bits_list), n_params
    )
    b_graph_ids = jnp.array(g_coord_b, dtype=jnp.int32)

    # ========================================================================
    # Type C compilation (π-pair)
    # Terms of the form (-1)^(ψ * φ) where ψ and φ are parities.
    # ========================================================================
    c_const_bits_a_list = []
    c_param_bits_a_list = []
    c_const_bits_b_list = []
    c_param_bits_b_list = []
    g_coord_c = []

    for i in range(num_graphs):
        graph = g_list[i]
        for p_set in graph.scalar.phasevars_pi_pair:
            # Parse set A
            const_bit_a = 1 if "1" in p_set[0] else 0
            param_bits_a = [0] * n_params
            for p in p_set[0]:
                if p != "1":
                    param_bits_a[char_to_idx[p]] = 1

            # Parse set B
            const_bit_b = 1 if "1" in p_set[1] else 0
            param_bits_b = [0] * n_params
            for p in p_set[1]:
                if p != "1":
                    param_bits_b[char_to_idx[p]] = 1

            g_coord_c.append(i)
            c_const_bits_a_list.append(const_bit_a)
            c_param_bits_a_list.append(param_bits_a)
            c_const_bits_b_list.append(const_bit_b)
            c_param_bits_b_list.append(param_bits_b)

    n_c_terms = len(c_const_bits_a_list)
    c_const_bits_a = jnp.array(c_const_bits_a_list, dtype=jnp.uint8)
    c_param_bits_a = jnp.array(c_param_bits_a_list, dtype=jnp.uint8).reshape(
        n_c_terms, n_params
    )
    c_const_bits_b = jnp.array(c_const_bits_b_list, dtype=jnp.uint8)
    c_param_bits_b = jnp.array(c_param_bits_b_list, dtype=jnp.uint8).reshape(
        n_c_terms, n_params
    )
    c_graph_ids = jnp.array(g_coord_c, dtype=jnp.int32)

    # ========================================================================
    # Type D compilation (phase-pair)
    # Terms of the form 1 + e^(i*alpha) + e^(i*beta) - e^(i*(alpha+beta))
    # ========================================================================
    d_const_alpha_list = []
    d_const_beta_list = []
    d_param_bits_a_list = []
    d_param_bits_b_list = []
    g_coord_d = []

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

            g_coord_d.append(i)
            d_const_alpha_list.append(const_alpha)
            d_const_beta_list.append(const_beta)
            d_param_bits_a_list.append(param_bits_a)
            d_param_bits_b_list.append(param_bits_b)

    n_d_terms = len(d_const_alpha_list)
    d_const_alpha = jnp.array(d_const_alpha_list, dtype=jnp.uint8)
    d_const_beta = jnp.array(d_const_beta_list, dtype=jnp.uint8)
    d_param_bits_a = jnp.array(d_param_bits_a_list, dtype=jnp.uint8).reshape(
        n_d_terms, n_params
    )
    d_param_bits_b = jnp.array(d_param_bits_b_list, dtype=jnp.uint8).reshape(
        n_d_terms, n_params
    )
    d_graph_ids = jnp.array(g_coord_d, dtype=jnp.int32)

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

    for g in g_list:
        dn = g.scalar.floatfactor.copy()

        p_sqrt2 = g.scalar.power2

        if p_sqrt2 % 2 != 0:
            p_sqrt2 -= 1
            dn *= DyadicNumber(k=0, a=0, b=1, c=0, d=1)

        assert p_sqrt2 % 2 == 0
        p_sqrt2 -= 2 * dn.k
        dn.k = 0

        power2.append(p_sqrt2 // 2)
        exact_floatfactor.append([dn.a, dn.b, dn.c, dn.d])

    return CompiledCircuit(
        num_graphs=num_graphs,
        n_params=n_params,
        a_const_phases=a_const_phases,
        a_param_bits=a_param_bits,
        a_graph_ids=a_graph_ids,
        b_term_types=b_term_types,
        b_param_bits=b_param_bits,
        b_graph_ids=b_graph_ids,
        c_const_bits_a=c_const_bits_a,
        c_param_bits_a=c_param_bits_a,
        c_const_bits_b=c_const_bits_b,
        c_param_bits_b=c_param_bits_b,
        c_graph_ids=c_graph_ids,
        d_const_alpha=d_const_alpha,
        d_const_beta=d_const_beta,
        d_param_bits_a=d_param_bits_a,
        d_param_bits_b=d_param_bits_b,
        d_graph_ids=d_graph_ids,
        phase_indices=phase_indices,
        has_approximate_floatfactors=has_approximate_floatfactors,
        approximate_floatfactors=approximate_floatfactors,
        power2=jnp.array(power2, dtype=jnp.int32),
        floatfactor=jnp.array(exact_floatfactor, dtype=jnp.int32),
    )
