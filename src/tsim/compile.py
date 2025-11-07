from typing import NamedTuple

import jax.numpy as jnp

from tsim.external.pyzx.graph.base import BaseGraph


class CompiledCircuit(NamedTuple):
    """JAX-compatible compiled circuit representation.

    All fields are static-shaped NumPy arrays, making them directly
    convertible to JAX arrays for GPU execution and JIT compilation.

    Constants and parameter-dependent data are separated for cleaner evaluation.
    """

    # Metadata
    num_graphs: int
    n_params: int

    # Type A/B: Node and Half-Pi Terms
    ab_term_types: jnp.ndarray  # shape: (n_ab_terms,), dtype: uint8, values: {2, 4, 6}
    ab_const_phases: jnp.ndarray  # shape: (n_ab_terms,), dtype: uint8, values: 0-7
    ab_param_bits: jnp.ndarray  # shape: (n_ab_terms, n_params), dtype: uint8
    ab_graph_ids: jnp.ndarray  # shape: (n_ab_terms,), dtype: int32

    # Type C: Pi-Pair Terms
    c_const_bits_a: jnp.ndarray  # shape: (n_c_terms,), dtype: uint8, values: {0, 1}
    c_param_bits_a: jnp.ndarray  # shape: (n_c_terms, n_params), dtype: uint8
    c_const_bits_b: jnp.ndarray  # shape: (n_c_terms,), dtype: uint8, values: {0, 1}
    c_param_bits_b: jnp.ndarray  # shape: (n_c_terms, n_params), dtype: uint8
    c_graph_ids: jnp.ndarray  # shape: (n_c_terms,), dtype: int32

    # Type D: Phase Pairs
    d_const_alpha: jnp.ndarray  # shape: (n_d_terms,), dtype: uint8, values: 0-7
    d_const_beta: jnp.ndarray  # shape: (n_d_terms,), dtype: uint8, values: 0-7
    d_param_bits_a: jnp.ndarray  # shape: (n_d_terms, n_params), dtype: uint8
    d_param_bits_b: jnp.ndarray  # shape: (n_d_terms, n_params), dtype: uint8
    d_graph_ids: jnp.ndarray  # shape: (n_d_terms,), dtype: int32

    # Static per-graph data
    # Shape: (num_graphs,)
    phase_indices: jnp.ndarray  # dtype: uint8 (values 0-7)
    power2: jnp.ndarray  # dtype: int32
    floatfactor: jnp.ndarray  # dtype: complex64


def compile_circuit(
    g_list: list[BaseGraph], n_params: int, chars: list[str]
) -> CompiledCircuit:
    """Compile ZX-graph list into JAX-compatible structure for fast evaluation.

    Args:
        g_list: List of ZX-graphs to compile
        n_params: Number of parameters (error bits + measurements/observables)
        chars: List of parameter names

    Returns:
        CompiledCircuit with all data in static-shaped arrays
    """
    num_graphs = len(g_list)
    char_to_idx = {char: i for i, char in enumerate(chars)}

    # ========================================================================
    # Type A/B compilation (phase-node/ half-π)
    # ========================================================================
    ab_term_types_list = []
    ab_const_phases_list = []
    ab_param_bits_list = []
    g_coord_ab = []

    for i in range(num_graphs):
        g_i = g_list[i]
        for term in range(len(g_i.scalar.phasenodevars)):
            bitstr = [0] * n_params
            for v in g_i.scalar.phasenodevars[term]:
                bitstr[char_to_idx[v]] = 1
            const_term = int(g_i.scalar.phasenodes[term] * 4)

            g_coord_ab.append(i)
            ab_term_types_list.append(4)
            ab_const_phases_list.append(const_term)
            ab_param_bits_list.append(bitstr)

        assert set(g_i.scalar.phasevars_halfpi.keys()) <= {1, 3}
        for j in [1, 3]:
            if j not in g_i.scalar.phasevars_halfpi:
                continue
            for term in range(len(g_i.scalar.phasevars_halfpi[j])):
                bitstr = [0] * n_params
                for v in g_i.scalar.phasevars_halfpi[j][term]:
                    bitstr[char_to_idx[v]] = 1
                ttype = int((j / 2) * 4)

                g_coord_ab.append(i)
                ab_term_types_list.append(ttype)
                ab_const_phases_list.append(0)
                ab_param_bits_list.append(bitstr)

    ab_term_types = (
        jnp.array(ab_term_types_list, dtype=jnp.uint8)
        if ab_term_types_list
        else jnp.zeros((0,), dtype=jnp.uint8)
    )
    ab_const_phases = (
        jnp.array(ab_const_phases_list, dtype=jnp.uint8)
        if ab_const_phases_list
        else jnp.zeros((0,), dtype=jnp.uint8)
    )
    ab_param_bits = (
        jnp.array(ab_param_bits_list, dtype=jnp.uint8)
        if ab_param_bits_list
        else jnp.zeros((0, n_params), dtype=jnp.uint8)
    )
    ab_graph_ids = (
        jnp.array(g_coord_ab, dtype=jnp.int32)
        if g_coord_ab
        else jnp.zeros((0,), dtype=jnp.int32)
    )

    # ========================================================================
    # Type C compilation (π-pair)
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

    c_const_bits_a = (
        jnp.array(c_const_bits_a_list, dtype=jnp.uint8)
        if c_const_bits_a_list
        else jnp.zeros((0,), dtype=jnp.uint8)
    )
    c_param_bits_a = (
        jnp.array(c_param_bits_a_list, dtype=jnp.uint8)
        if c_param_bits_a_list
        else jnp.zeros((0, n_params), dtype=jnp.uint8)
    )
    c_const_bits_b = (
        jnp.array(c_const_bits_b_list, dtype=jnp.uint8)
        if c_const_bits_b_list
        else jnp.zeros((0,), dtype=jnp.uint8)
    )
    c_param_bits_b = (
        jnp.array(c_param_bits_b_list, dtype=jnp.uint8)
        if c_param_bits_b_list
        else jnp.zeros((0, n_params), dtype=jnp.uint8)
    )
    c_graph_ids = (
        jnp.array(g_coord_c, dtype=jnp.int32)
        if g_coord_c
        else jnp.zeros((0,), dtype=jnp.int32)
    )

    # ========================================================================
    # Type D compilation (phase-pair)
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

    d_const_alpha = (
        jnp.array(d_const_alpha_list, dtype=jnp.uint8)
        if d_const_alpha_list
        else jnp.zeros((0,), dtype=jnp.uint8)
    )
    d_const_beta = (
        jnp.array(d_const_beta_list, dtype=jnp.uint8)
        if d_const_beta_list
        else jnp.zeros((0,), dtype=jnp.uint8)
    )
    d_param_bits_a = (
        jnp.array(d_param_bits_a_list, dtype=jnp.uint8)
        if d_param_bits_a_list
        else jnp.zeros((0, n_params), dtype=jnp.uint8)
    )
    d_param_bits_b = (
        jnp.array(d_param_bits_b_list, dtype=jnp.uint8)
        if d_param_bits_b_list
        else jnp.zeros((0, n_params), dtype=jnp.uint8)
    )
    d_graph_ids = (
        jnp.array(g_coord_d, dtype=jnp.int32)
        if g_coord_d
        else jnp.zeros((0,), dtype=jnp.int32)
    )

    # ========================================================================
    # Static data
    # ========================================================================
    phase_indices = jnp.array(
        [int(g.scalar.phase * 4) for g in g_list], dtype=jnp.uint8
    )
    power2 = jnp.array([g.scalar.power2 for g in g_list], dtype=jnp.int32)
    floatfactor = jnp.array([g.scalar.floatfactor for g in g_list], dtype=jnp.complex64)

    return CompiledCircuit(
        num_graphs=num_graphs,
        n_params=n_params,
        ab_term_types=ab_term_types,
        ab_const_phases=ab_const_phases,
        ab_param_bits=ab_param_bits,
        ab_graph_ids=ab_graph_ids,
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
        power2=power2,
        floatfactor=floatfactor,
    )
