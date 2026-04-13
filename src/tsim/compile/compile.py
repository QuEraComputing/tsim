"""Compilation of ZX graphs into JAX-compatible data structures."""

from collections import defaultdict
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from pyzx_param.graph.base import BaseGraph
from pyzx_param.graph.scalar import DyadicNumber


class NodePhases(eqx.Module):
    """Product of ``1 + exp(i·(α + ⊕params)·π)`` terms, one factor per stored term.

    Padded slots use ``0`` for both ``phases`` and ``params``; the evaluator masks
    padded slots to the multiplicative identity using ``counts``.

    Shapes are ``(num_graphs, max_terms)`` except ``params`` which is
    ``(num_graphs, max_terms, n_params)``.
    """

    phases: Array  # uint8, values 0-7 (the constant offset α, as ``4·α``)
    params: Array  # uint8, parameter parity bitmasks
    counts: Array  # int32, number of real (non-padded) terms per graph


class HalfPiPhases(eqx.Module):
    """Sum of ``exp(i·j·π·⊕params / 2)`` terms with ``j ∈ {1, 3}``.

    Terms sharing a parameter bitstring have been combined to a single stored
    coefficient ``j' ∈ {1, 2, 3}`` (see ``_compile_halfpi_phases``). Coefficients
    are stored in eighth-turn units — i.e. as ``2·j'`` — so the evaluator can
    reuse the ``ω = e^(iπ/4)`` phase table.

    Padded slots use ``0`` (the additive identity for phase sums), so padded
    entries contribute nothing to the summed exponent.

    Shapes are ``(num_graphs, max_terms)`` except ``params`` which is
    ``(num_graphs, max_terms, n_params)``.
    """

    coeffs: Array  # uint8, values in {0, 2, 4, 6}  (= 2·j', with 0 = padding)
    params: Array  # uint8, parameter parity bitmasks


class PiProducts(eqx.Module):
    """Product of ``(-1)^(ψ · φ)`` terms, with ψ and φ each a parity expression.

    Each side is encoded as a constant bit plus a parameter bitmask. Padded slots
    use ``0`` everywhere; a padded term contributes ``(-1)^0 = 1`` to the product.

    Shapes are ``(num_graphs, max_terms)`` except ``*_params`` which are
    ``(num_graphs, max_terms, n_params)``.
    """

    psi_const: Array  # uint8, values {0, 1}
    psi_params: Array  # uint8, parameter parity bitmask for ψ
    phi_const: Array  # uint8, values {0, 1}
    phi_params: Array  # uint8, parameter parity bitmask for φ


class PhasePairs(eqx.Module):
    """Product of ``1 + e^(iα) + e^(iβ) − e^(i(α+β))`` terms.

    Each of ``α`` and ``β`` combines a constant phase with a parameter parity.
    Padded slots use ``0`` and are masked to the multiplicative identity using
    ``counts``.

    Shapes are ``(num_graphs, max_terms)`` except ``*_params`` which are
    ``(num_graphs, max_terms, n_params)``.
    """

    alpha: Array  # uint8, values 0-7 (constant offset of α, as ``4·α``)
    alpha_params: Array  # uint8, parameter parity bitmask for α
    beta: Array  # uint8, values 0-7 (constant offset of β, as ``4·β``)
    beta_params: Array  # uint8, parameter parity bitmask for β
    counts: Array  # int32, number of real (non-padded) terms per graph


class CompiledScalarGraphs(eqx.Module):
    """JAX-compatible compiled representation of a list of scalar ZX graphs.

    The scalar for each graph is a product of four term families, multiplied by
    a global phase and a floatfactor. All arrays are static-shaped so the whole
    struct can be traced under ``jax.jit``.
    """

    num_graphs: int
    n_params: int

    node_phases: NodePhases
    halfpi_phases: HalfPiPhases
    pi_products: PiProducts
    phase_pairs: PhasePairs

    # Static per-graph data
    phase_indices: Array  # shape: (num_graphs,), dtype: uint8 (values 0-7)
    has_approximate_floatfactors: bool = eqx.field(static=True)
    approximate_floatfactors: Array  # shape: (num_graphs,), dtype: complex64
    power2: Array  # shape: (num_graphs,), dtype: int32
    floatfactor: Array  # shape: (num_graphs, 4), dtype: int32


def _compile_node_phases(
    g_list: list[BaseGraph], char_to_idx: dict[str, int], n_params: int
) -> NodePhases:
    """Compile ``1 + exp(i·(α + ⊕params)·π)`` terms into padded arrays.

    ``α`` is a rational constant phase with denominator in ``{1, 2, 4}``, stored
    as ``4·α``. ``⊕params`` is the XOR of a subset of parameter bits.

    Args:
        g_list: List of scalar ZX graphs.
        char_to_idx: Mapping from parameter name to column index.
        n_params: Total number of parameters.

    Returns:
        A ``NodePhases`` with term arrays padded to the max term count across
        ``g_list``.

    """
    num_graphs = len(g_list)
    terms_per_graph: list[list[tuple[int, list[int]]]] = [[] for _ in range(num_graphs)]

    for i, g_i in enumerate(g_list):
        for term in range(len(g_i.scalar.phasenodevars)):
            bitstr = [0] * n_params
            for v in g_i.scalar.phasenodevars[term]:
                bitstr[char_to_idx[v]] = 1
            assert g_i.scalar.phasenodes[term].denominator in [1, 2, 4]
            const_term = int(g_i.scalar.phasenodes[term] * 4)  # type: ignore[arg-type]
            terms_per_graph[i].append((const_term, bitstr))

    counts = np.array([len(terms) for terms in terms_per_graph], dtype=np.int32)
    max_terms = int(counts.max()) if counts.size else 0

    phases = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    params = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)

    for i, terms in enumerate(terms_per_graph):
        for j, (const_phase, param_bit) in enumerate(terms):
            phases[i, j] = const_phase
            params[i, j] = param_bit

    return NodePhases(
        phases=jnp.array(phases),
        params=jnp.array(params),
        counts=jnp.array(counts, dtype=jnp.int32),
    )


def _compile_halfpi_phases(
    g_list: list[BaseGraph], char_to_idx: dict[str, int], n_params: int
) -> HalfPiPhases:
    """Compile ``exp(i·j·π·⊕params / 2)`` terms (``j ∈ {1, 3}``) into padded arrays.

    Terms sharing the same parameter bitstring are combined into a single
    coefficient ``j' = (Σ j) mod 4 ∈ {0, 1, 2, 3}``; the ``j' = 0`` case is
    dropped. The surviving ``j'`` is stored in eighth-turn units — as
    ``2·j' ∈ {2, 4, 6}`` — so the stored coefficient feeds directly into the
    ``ω = e^(iπ/4)`` phase table used by the evaluator. Padded slots use 0.

    Args:
        g_list: List of scalar ZX graphs.
        char_to_idx: Mapping from parameter name to column index.
        n_params: Total number of parameters.

    Returns:
        A ``HalfPiPhases``. Padded slots contain 0, the additive identity for
        phase sums.

    """
    num_graphs = len(g_list)
    terms_per_graph: list[list[tuple[int, list[int]]]] = [[] for _ in range(num_graphs)]

    for i, g_i in enumerate(g_list):
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
            terms_per_graph[i].append((combined_j * 2, list(bitstr_key)))

    max_terms = max((len(terms) for terms in terms_per_graph), default=0)

    coeffs = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    params = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)

    for i, terms in enumerate(terms_per_graph):
        for j, (coeff, param_bit) in enumerate(terms):
            coeffs[i, j] = coeff
            params[i, j] = param_bit

    return HalfPiPhases(coeffs=jnp.array(coeffs), params=jnp.array(params))


def _compile_pi_products(
    g_list: list[BaseGraph], char_to_idx: dict[str, int], n_params: int
) -> PiProducts:
    """Compile ``(-1)^(ψ · φ)`` terms into padded arrays.

    Each of ψ and φ is a parity over parameters with a boolean constant offset.

    Args:
        g_list: List of scalar ZX graphs.
        char_to_idx: Mapping from parameter name to column index.
        n_params: Total number of parameters.

    Returns:
        A ``PiProducts``. Padded slots contain 0 on every field so they
        contribute ``(-1)^0 = 1`` to the product.

    """
    num_graphs = len(g_list)
    terms_per_graph: list[list[tuple[int, list[int], int, list[int]]]] = [
        [] for _ in range(num_graphs)
    ]

    for i, graph in enumerate(g_list):
        for p_set in graph.scalar.phasevars_pi_pair:
            psi_const = 1 if "1" in p_set[0] else 0
            psi_params = [0] * n_params
            for p in p_set[0]:
                if p != "1":
                    psi_params[char_to_idx[p]] = 1

            phi_const = 1 if "1" in p_set[1] else 0
            phi_params = [0] * n_params
            for p in p_set[1]:
                if p != "1":
                    phi_params[char_to_idx[p]] = 1

            terms_per_graph[i].append((psi_const, psi_params, phi_const, phi_params))

    max_terms = max((len(terms) for terms in terms_per_graph), default=0)

    psi_const_arr = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    psi_params_arr = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)
    phi_const_arr = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    phi_params_arr = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)

    for i, terms in enumerate(terms_per_graph):
        for j, (psi_c, psi_p, phi_c, phi_p) in enumerate(terms):
            psi_const_arr[i, j] = psi_c
            psi_params_arr[i, j] = psi_p
            phi_const_arr[i, j] = phi_c
            phi_params_arr[i, j] = phi_p

    return PiProducts(
        psi_const=jnp.array(psi_const_arr),
        psi_params=jnp.array(psi_params_arr),
        phi_const=jnp.array(phi_const_arr),
        phi_params=jnp.array(phi_params_arr),
    )


def _compile_phase_pairs(
    g_list: list[BaseGraph], char_to_idx: dict[str, int], n_params: int
) -> PhasePairs:
    """Compile ``1 + e^(iα) + e^(iβ) − e^(i(α+β))`` terms into padded arrays.

    ``α`` and ``β`` each combine a constant phase (stored as ``4·α``, ``4·β``)
    with a parameter parity.

    Args:
        g_list: List of scalar ZX graphs.
        char_to_idx: Mapping from parameter name to column index.
        n_params: Total number of parameters.

    Returns:
        A ``PhasePairs`` with term arrays padded to the max term count across
        ``g_list``. Padded slots are masked to the multiplicative identity at
        evaluation time using ``counts``.

    """
    num_graphs = len(g_list)
    terms_per_graph: list[list[tuple[int, int, list[int], list[int]]]] = [
        [] for _ in range(num_graphs)
    ]

    for i, graph in enumerate(g_list):
        for pp in range(len(graph.scalar.phasepairs)):
            alpha_params = [0] * n_params
            for v in graph.scalar.phasepairs[pp].paramsA:
                alpha_params[char_to_idx[v]] = 1

            beta_params = [0] * n_params
            for v in graph.scalar.phasepairs[pp].paramsB:
                beta_params[char_to_idx[v]] = 1

            const_alpha = int(graph.scalar.phasepairs[pp].alpha)
            const_beta = int(graph.scalar.phasepairs[pp].beta)

            terms_per_graph[i].append(
                (const_alpha, const_beta, alpha_params, beta_params)
            )

    counts = np.array([len(terms) for terms in terms_per_graph], dtype=np.int32)
    max_terms = int(counts.max()) if counts.size else 0

    alpha = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    beta = np.zeros((num_graphs, max_terms), dtype=np.uint8)
    alpha_params_arr = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)
    beta_params_arr = np.zeros((num_graphs, max_terms, n_params), dtype=np.uint8)

    for i, terms in enumerate(terms_per_graph):
        for j, (ca, cb, pa, pb) in enumerate(terms):
            alpha[i, j] = ca
            beta[i, j] = cb
            alpha_params_arr[i, j] = pa
            beta_params_arr[i, j] = pb

    return PhasePairs(
        alpha=jnp.array(alpha),
        alpha_params=jnp.array(alpha_params_arr),
        beta=jnp.array(beta),
        beta_params=jnp.array(beta_params_arr),
        counts=jnp.array(counts, dtype=jnp.int32),
    )


def _compile_static_scalar_data(
    g_list: list[BaseGraph],
) -> tuple[bool, Array, Array, Array, Array]:
    """Compile the per-graph static scalar data.

    For each graph this extracts:

    - the phase index (``4·phase`` as an integer in ``0..7``), folding any phase
      whose denominator is not in ``{1, 2, 4}`` into ``approximate_floatfactor``,
    - an approximate complex ``floatfactor`` (mutated above) and whether any
      graph carries a non-trivial one,
    - an exact dyadic ``floatfactor`` decomposition ``(a, b, c, d)`` together
      with the accompanying power of 2.

    Side effect: mutates ``g.scalar.phase`` and ``g.scalar.approximate_floatfactor``
    on graphs whose phase denominator is not in ``{1, 2, 4}``.

    Args:
        g_list: List of scalar ZX graphs.

    Returns:
        Tuple ``(has_approximate_floatfactors, approximate_floatfactors,
        phase_indices, power2, floatfactor)`` as JAX arrays.

    """
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

    return (
        has_approximate_floatfactors,
        approximate_floatfactors,
        phase_indices,
        jnp.array(power2, dtype=jnp.int32),
        jnp.array(exact_floatfactor, dtype=jnp.int32),
    )


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
        n_vertices = len(list(g.vertices()))
        if n_vertices != 0:
            raise ValueError(
                f"Only scalar graphs can be compiled but graph {i} has {n_vertices} vertices"
            )

    g_list = [g for g in g_list if not g.scalar.is_zero]

    n_params = len(params)
    num_graphs = len(g_list)
    char_to_idx = {char: i for i, char in enumerate(params)}

    node_phases = _compile_node_phases(g_list, char_to_idx, n_params)
    halfpi_phases = _compile_halfpi_phases(g_list, char_to_idx, n_params)
    pi_products = _compile_pi_products(g_list, char_to_idx, n_params)
    phase_pairs = _compile_phase_pairs(g_list, char_to_idx, n_params)
    (
        has_approximate_floatfactors,
        approximate_floatfactors,
        phase_indices,
        power2,
        floatfactor,
    ) = _compile_static_scalar_data(g_list)

    return CompiledScalarGraphs(
        num_graphs=num_graphs,
        n_params=n_params,
        node_phases=node_phases,
        halfpi_phases=halfpi_phases,
        pi_products=pi_products,
        phase_pairs=phase_pairs,
        phase_indices=phase_indices,
        has_approximate_floatfactors=has_approximate_floatfactors,
        approximate_floatfactors=approximate_floatfactors,
        power2=power2,
        floatfactor=floatfactor,
    )
