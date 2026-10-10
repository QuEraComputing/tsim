"""Compilation of ZX graphs into JAX-compatible data structures."""

import cmath
import math
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from pyzx_param.graph.base import BaseGraph
from pyzx_param.graph.scalar import DyadicNumber
from zx_param import Scalar

from tsim.compile.terms import (
    FloatTerms,
    HalfPiPhases,
    NodePhases,
    PhasePairs,
    PiProducts,
    ScalarPrefactor,
)


class CompiledScalarGraphs(eqx.Module):
    """JAX-compatible compiled representation of a list of scalar ZX graphs.

    The scalar for each graph is a product of four term families, multiplied by
    a per-graph ``ScalarPrefactor`` (global phase, floatfactor, ``2^power2``,
    optional approximate complex floatfactor). All arrays are static-shaped so
    the whole struct can be traced under ``jax.jit``.
    """

    num_graphs: int
    n_params: int

    node_phases: NodePhases
    halfpi_phases: HalfPiPhases
    pi_products: PiProducts
    phase_pairs: PhasePairs
    float_terms: FloatTerms
    prefactor: ScalarPrefactor
    has_float_terms: bool = eqx.field(static=True, default=False)


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
            phases[i, j] = const_phase % 8
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
            alpha[i, j] = ca % 8
            beta[i, j] = cb % 8
            alpha_params_arr[i, j] = pa
            beta_params_arr[i, j] = pb

    return PhasePairs(
        alpha=jnp.array(alpha),
        alpha_params=jnp.array(alpha_params_arr),
        beta=jnp.array(beta),
        beta_params=jnp.array(beta_params_arr),
        counts=jnp.array(counts, dtype=jnp.int32),
    )


def _compile_prefactor(g_list: list[BaseGraph]) -> ScalarPrefactor:
    """Compile the per-graph static scalar prefactor.

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
        A ``ScalarPrefactor`` holding the per-graph static scalar data.

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
        [int(float(g.scalar.phase) * 4) % 8 for g in g_list], dtype=jnp.uint8
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

    return ScalarPrefactor(
        phase_indices=phase_indices,
        floatfactor=jnp.array(exact_floatfactor, dtype=jnp.int32).reshape(-1, 4),
        power2=jnp.array(power2, dtype=jnp.int32),
        approximate_floatfactors=approximate_floatfactors,
        has_approximate_floatfactors=has_approximate_floatfactors,
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
        if g.scalar.phasevars_pi and not g.scalar.is_zero:
            raise NotImplementedError(
                f"compile_scalar_graphs does not support Scalar.phasevars_pi "
                f"(graph {i} has phasevars_pi={sorted(g.scalar.phasevars_pi)!r})"
            )

    g_list = [g for g in g_list if not g.scalar.is_zero]

    n_params = len(params)
    num_graphs = len(g_list)
    char_to_idx = {char: i for i, char in enumerate(params)}

    return CompiledScalarGraphs(
        num_graphs=num_graphs,
        n_params=n_params,
        node_phases=_compile_node_phases(g_list, char_to_idx, n_params),
        halfpi_phases=_compile_halfpi_phases(g_list, char_to_idx, n_params),
        pi_products=_compile_pi_products(g_list, char_to_idx, n_params),
        phase_pairs=_compile_phase_pairs(g_list, char_to_idx, n_params),
        float_terms=_empty_float_terms(num_graphs, n_params),
        prefactor=_compile_prefactor(g_list),
    )


def _empty_float_terms(num_graphs: int, n_params: int) -> FloatTerms:
    """Return a ``FloatTerms`` with no terms."""
    return FloatTerms(
        table=jnp.ones((num_graphs, 0, 4), dtype=jnp.complex64),
        a_params=jnp.zeros((num_graphs, 0, n_params), dtype=jnp.uint8),
        b_params=jnp.zeros((num_graphs, 0, n_params), dtype=jnp.uint8),
    )


def _cis(alpha: Fraction | float) -> complex:
    """Return ``e^{iπ·alpha}``."""
    return cmath.exp(1j * math.pi * float(alpha))


def _eighths(alpha: Fraction) -> int | None:
    """Return ``4·alpha mod 8`` if ``alpha`` is a multiple of 1/4, else None."""
    scaled = Fraction(alpha) * 4
    if scaled.denominator != 1:
        return None
    return int(scaled) % 8


def _phase_pair_value(alpha: Fraction, beta: Fraction, pa: int, pb: int) -> complex:
    """Return ``1 + e^{iπ(α+pa)} + e^{iπ(β+pb)} − e^{iπ(α+β+pa+pb)}``."""
    return 1 + _cis(alpha + pa) + _cis(beta + pb) - _cis(alpha + beta + pa + pb)


@dataclass
class _SplitScalar:
    """One scalar's terms, sorted into the compiled families."""

    nodes: list[tuple[int, list[int]]] = field(default_factory=list)
    phases: dict[tuple[int, ...], int] = field(default_factory=dict)
    pi_pairs: list[tuple[list[int], list[int]]] = field(default_factory=list)
    phase_pairs: list[tuple[int, int, list[int], list[int]]] = field(
        default_factory=list
    )
    floats: list[tuple[list[complex], list[int], list[int]]] = field(
        default_factory=list
    )
    float_factor: complex = 1.0
    phase_index: int = 0
    coeffs: tuple[int, int, int, int] = (1, 0, 0, 0)
    power2: int = 0


def _split_scalar(s: Scalar, column: dict[int, int], n_params: int) -> _SplitScalar:
    """Sort the terms of ``s`` into exact families, float terms and float factor."""

    def mask(params: frozenset[int]) -> list[int]:
        bits = [0] * n_params
        for p in params:
            bits[column[p]] = 1
        return bits

    split = _SplitScalar()
    for n in s.nodes:
        k = _eighths(n.alpha)
        if k is not None:
            split.nodes.append((k, mask(n.a)))
        elif n.a:
            v0, v1 = 1 + _cis(n.alpha), 1 - _cis(n.alpha)
            split.floats.append(([v0, v1, v0, v1], mask(n.a), [0] * n_params))
        else:
            split.float_factor *= 1 + _cis(n.alpha)
    for h in s.phases:
        if not h.a:
            continue
        k = _eighths(h.alpha)
        if k is not None:
            key = tuple(mask(h.a))
            split.phases[key] = (split.phases.get(key, 0) + k) % 8
        else:
            w = _cis(h.alpha)
            split.floats.append(([1, w, 1, w], mask(h.a), [0] * n_params))
    for p in s.pi_pairs:
        if p.a and p.b:
            split.pi_pairs.append((mask(p.a), mask(p.b)))
    for q in s.phase_pairs:
        ka, kb = _eighths(q.alpha), _eighths(q.beta)
        if ka is not None and kb is not None:
            split.phase_pairs.append((ka, kb, mask(q.a), mask(q.b)))
            continue
        table = [
            _phase_pair_value(q.alpha, q.beta, pa, pb) for pb in (0, 1) for pa in (0, 1)
        ]
        if q.a or q.b:
            split.floats.append((table, mask(q.a), mask(q.b)))
        else:
            split.float_factor *= table[0]
    k = _eighths(s.phase)
    if k is not None:
        split.phase_index = k
    else:
        split.float_factor *= _cis(s.phase)
    base = s.base
    split.coeffs = (base.a, base.b, base.c, base.d)
    split.power2 = base.power
    return split


def _node_phase_family(split: list[_SplitScalar], n_params: int) -> NodePhases:
    num_graphs = len(split)
    width = max((len(s.nodes) for s in split), default=0)
    phases = np.zeros((num_graphs, width), dtype=np.uint8)
    params = np.zeros((num_graphs, width, n_params), dtype=np.uint8)
    counts = np.zeros(num_graphs, dtype=np.int32)
    for i, s in enumerate(split):
        counts[i] = len(s.nodes)
        for j, (k, bits) in enumerate(s.nodes):
            phases[i, j] = k
            params[i, j] = bits
    return NodePhases(
        phases=jnp.array(phases), params=jnp.array(params), counts=jnp.array(counts)
    )


def _phase_term_family(split: list[_SplitScalar], n_params: int) -> HalfPiPhases:
    rows = [[(k, bits) for bits, k in s.phases.items() if k] for s in split]
    width = max((len(r) for r in rows), default=0)
    coeffs = np.zeros((len(split), width), dtype=np.uint8)
    params = np.zeros((len(split), width, n_params), dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, (k, bits) in enumerate(row):
            coeffs[i, j] = k
            params[i, j] = bits
    return HalfPiPhases(coeffs=jnp.array(coeffs), params=jnp.array(params))


def _pi_product_family(split: list[_SplitScalar], n_params: int) -> PiProducts:
    width = max((len(s.pi_pairs) for s in split), default=0)
    zeros = np.zeros((len(split), width), dtype=np.uint8)
    psi = np.zeros((len(split), width, n_params), dtype=np.uint8)
    phi = np.zeros((len(split), width, n_params), dtype=np.uint8)
    for i, s in enumerate(split):
        for j, (a, b) in enumerate(s.pi_pairs):
            psi[i, j] = a
            phi[i, j] = b
    return PiProducts(
        psi_const=jnp.array(zeros),
        psi_params=jnp.array(psi),
        phi_const=jnp.array(zeros),
        phi_params=jnp.array(phi),
    )


def _phase_pair_family(split: list[_SplitScalar], n_params: int) -> PhasePairs:
    num_graphs = len(split)
    width = max((len(s.phase_pairs) for s in split), default=0)
    alpha = np.zeros((num_graphs, width), dtype=np.uint8)
    beta = np.zeros((num_graphs, width), dtype=np.uint8)
    alpha_params = np.zeros((num_graphs, width, n_params), dtype=np.uint8)
    beta_params = np.zeros((num_graphs, width, n_params), dtype=np.uint8)
    counts = np.zeros(num_graphs, dtype=np.int32)
    for i, s in enumerate(split):
        counts[i] = len(s.phase_pairs)
        for j, (ka, kb, a, b) in enumerate(s.phase_pairs):
            alpha[i, j], beta[i, j] = ka, kb
            alpha_params[i, j], beta_params[i, j] = a, b
    return PhasePairs(
        alpha=jnp.array(alpha),
        alpha_params=jnp.array(alpha_params),
        beta=jnp.array(beta),
        beta_params=jnp.array(beta_params),
        counts=jnp.array(counts),
    )


def _float_term_family(split: list[_SplitScalar], n_params: int) -> FloatTerms:
    num_graphs = len(split)
    width = max((len(s.floats) for s in split), default=0)
    table = np.ones((num_graphs, width, 4), dtype=np.complex64)
    a_params = np.zeros((num_graphs, width, n_params), dtype=np.uint8)
    b_params = np.zeros((num_graphs, width, n_params), dtype=np.uint8)
    for i, s in enumerate(split):
        for j, (values, a, b) in enumerate(s.floats):
            table[i, j] = values
            a_params[i, j], b_params[i, j] = a, b
    return FloatTerms(
        table=jnp.array(table),
        a_params=jnp.array(a_params),
        b_params=jnp.array(b_params),
    )


def _scalar_prefactor(split: list[_SplitScalar]) -> ScalarPrefactor:
    coeffs = [s.coeffs for s in split]
    if any(abs(c) >= 2**31 for row in coeffs for c in row):
        raise OverflowError("a scalar's Dyadic coefficients do not fit in int32")
    approximate = [s.float_factor for s in split]
    return ScalarPrefactor(
        phase_indices=jnp.array([s.phase_index for s in split], dtype=jnp.uint8),
        floatfactor=jnp.array(coeffs, dtype=jnp.int32).reshape(-1, 4),
        power2=jnp.array([s.power2 for s in split], dtype=jnp.int32),
        approximate_floatfactors=jnp.array(approximate, dtype=jnp.complex64),
        has_approximate_floatfactors=any(f != 1 for f in approximate),
    )


def compile_scalars(
    scalars: list[Scalar], param_ids: list[int]
) -> CompiledScalarGraphs:
    """Compile exact scalars, one per stabilizer term, into JAX arrays.

    Column ``k`` of the parameter values the result is evaluated on holds the bit
    of parameter ``param_ids[k]``. Zero scalars are dropped. Angles that are
    multiples of π/4 stay exact; other angles go into each term's float factor
    (terms without parameters) or into the float term family.

    Raises:
        KeyError: A term uses a parameter that is not in ``param_ids``.
        OverflowError: A Dyadic coefficient does not fit in int32.

    """
    column = {p: k for k, p in enumerate(param_ids)}
    n_params = len(param_ids)
    split = [_split_scalar(s, column, n_params) for s in scalars if not s.is_zero]
    return CompiledScalarGraphs(
        num_graphs=len(split),
        n_params=n_params,
        node_phases=_node_phase_family(split, n_params),
        halfpi_phases=_phase_term_family(split, n_params),
        pi_products=_pi_product_family(split, n_params),
        phase_pairs=_phase_pair_family(split, n_params),
        float_terms=_float_term_family(split, n_params),
        prefactor=_scalar_prefactor(split),
        has_float_terms=any(s.floats for s in split),
    )
