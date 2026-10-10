"""Compilation of ZX graphs into JAX-compatible data structures."""

import cmath
import math
from dataclasses import dataclass, field
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from zx_param import Scalar

from tsim.compile.terms import (
    FloatTerms,
    NodePhases,
    PhasePairs,
    PhaseTerms,
    PiProducts,
    ScalarPrefactor,
)


class CompiledScalarGraphs(eqx.Module):
    """JAX-compatible compiled representation of a list of scalar ZX graphs.

    The scalar for each graph is a product of four exact term families and the
    floating-point ``FloatTerms`` family, multiplied by a per-graph
    ``ScalarPrefactor`` (global phase, floatfactor, ``2^power2``, optional
    approximate complex floatfactor). All arrays are static-shaped so
    the whole struct can be traced under ``jax.jit``.
    """

    num_graphs: int
    n_params: int

    node_phases: NodePhases
    phase_terms: PhaseTerms
    pi_products: PiProducts
    phase_pairs: PhasePairs
    float_terms: FloatTerms
    prefactor: ScalarPrefactor
    has_float_terms: bool = eqx.field(static=True, default=False)


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


def _phase_term_family(split: list[_SplitScalar], n_params: int) -> PhaseTerms:
    rows = [[(k, bits) for bits, k in s.phases.items() if k] for s in split]
    width = max((len(r) for r in rows), default=0)
    coeffs = np.zeros((len(split), width), dtype=np.uint8)
    params = np.zeros((len(split), width, n_params), dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, (k, bits) in enumerate(row):
            coeffs[i, j] = k
            params[i, j] = bits
    return PhaseTerms(coeffs=jnp.array(coeffs), params=jnp.array(params))


def _pi_product_family(split: list[_SplitScalar], n_params: int) -> PiProducts:
    width = max((len(s.pi_pairs) for s in split), default=0)
    psi = np.zeros((len(split), width, n_params), dtype=np.uint8)
    phi = np.zeros((len(split), width, n_params), dtype=np.uint8)
    for i, s in enumerate(split):
        for j, (a, b) in enumerate(s.pi_pairs):
            psi[i, j] = a
            phi[i, j] = b
    return PiProducts(psi_params=jnp.array(psi), phi_params=jnp.array(phi))


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
        phase_terms=_phase_term_family(split, n_params),
        pi_products=_pi_product_family(split, n_params),
        phase_pairs=_phase_pair_family(split, n_params),
        float_terms=_float_term_family(split, n_params),
        prefactor=_scalar_prefactor(split),
        has_float_terms=any(s.floats for s in split),
    )
