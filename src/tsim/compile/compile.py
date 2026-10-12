"""Compilation of ZX graphs into JAX-compatible data structures."""

import cmath
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from zxparam import Scalar
from zxparam.dyadic import omega_exponent

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

    @property
    def has_float_terms(self) -> bool:
        """Whether any graph has a float term (a static shape, so safe under jit)."""
        return self.float_terms.table.shape[1] > 0


def _cis(alpha: Fraction | float) -> complex:
    """Return ``e^{iπ·alpha}``."""
    return cmath.exp(1j * math.pi * float(alpha))


def _phase_pair_value(alpha: Fraction, beta: Fraction, pa: int, pb: int) -> complex:
    """Return ``1 + e^{iπ(α+pa)} + e^{iπ(β+pb)} − e^{iπ(α+β+pa+pb)}``."""
    return 1 + _cis(alpha + pa) + _cis(beta + pb) - _cis(alpha + beta + pa + pb)


@dataclass
class _SplitScalar:
    """One scalar's terms, sorted into the compiled families."""

    nodes: list[tuple[int, list[int]]] = field(default_factory=list)
    phases: list[tuple[int, list[int]]] = field(default_factory=list)
    pi_pairs: list[tuple[list[int], list[int]]] = field(default_factory=list)
    phase_pairs: list[tuple[int, int, list[int], list[int]]] = field(
        default_factory=list
    )
    floats: list[tuple[list[complex], list[int], list[int]]] = field(
        default_factory=list
    )
    float_factor: complex = 1.0
    coeffs: tuple[int, int, int, int] = (1, 0, 0, 0)
    power2: int = 0


def _split_scalar(s: Scalar, column: dict[int, int], n_params: int) -> _SplitScalar:
    """Sort the terms of ``s`` into exact families, float terms and float factor.

    ``s`` must be ``Scalar.simplified``: each phase term has its own nonempty
    parameter set, no pi-pair has an empty side, and a global phase that is a
    multiple of π/4 is already in ``base``.
    """

    def mask(params: frozenset[int]) -> list[int]:
        bits = [0] * n_params
        for p in params:
            bits[column[p]] = 1
        return bits

    split = _SplitScalar()
    for n in s.nodes:
        k = omega_exponent(n.alpha)
        if k is not None:
            split.nodes.append((k, mask(n.a)))
        elif n.a:
            v0, v1 = 1 + _cis(n.alpha), 1 - _cis(n.alpha)
            split.floats.append(([v0, v1, v0, v1], mask(n.a), [0] * n_params))
        else:
            split.float_factor *= 1 + _cis(n.alpha)
    for h in s.phases:
        k = omega_exponent(h.alpha)
        if k is not None:
            split.phases.append((k, mask(h.a)))
        else:
            w = _cis(h.alpha)
            split.floats.append(([1, w, 1, w], mask(h.a), [0] * n_params))
    for p in s.pi_pairs:
        split.pi_pairs.append((mask(p.a), mask(p.b)))
    for q in s.phase_pairs:
        ka, kb = omega_exponent(q.alpha), omega_exponent(q.beta)
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
    if s.phase:
        split.float_factor *= _cis(s.phase)
    base = s.base
    split.coeffs = (base.a, base.b, base.c, base.d)
    split.power2 = base.power
    return split


def _padded(
    rows: list[list[Any]], shape: tuple[int, ...], dtype: Any, fill: Any = 0
) -> jax.Array:
    """Stack per-graph rows into a ``(num_graphs, max_len, *shape)`` array.

    Short rows are padded with ``fill``.
    """
    width = max((len(row) for row in rows), default=0)
    out = np.full((len(rows), width, *shape), fill, dtype=dtype)
    for i, row in enumerate(rows):
        if row:
            out[i, : len(row)] = row
    return jnp.array(out)


def _scalar_prefactor(split: list[_SplitScalar]) -> ScalarPrefactor:
    coeffs = [s.coeffs for s in split]
    if any(abs(c) >= 2**31 for row in coeffs for c in row):
        raise OverflowError("a scalar's Dyadic coefficients do not fit in int32")
    approximate = [s.float_factor for s in split]
    return ScalarPrefactor(
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
    of parameter ``param_ids[k]``. Each scalar is ``Scalar.simplified`` first,
    and zero scalars are dropped. Angles that are multiples of π/4 stay exact;
    other angles go into each term's float factor (terms without parameters) or
    into the float term family.

    Raises:
        KeyError: A term uses a parameter that is not in ``param_ids``.
        OverflowError: A Dyadic coefficient does not fit in int32.

    """
    column = {p: k for k, p in enumerate(param_ids)}
    n_params = len(param_ids)
    simplified = [s.simplified() for s in scalars]
    split = [_split_scalar(s, column, n_params) for s in simplified if not s.is_zero]

    def padded_field(
        family: str,
        index: int,
        shape: tuple[int, ...] = (),
        dtype: Any = np.uint8,
        fill: Any = 0,
    ) -> jax.Array:
        rows = [[term[index] for term in getattr(s, family)] for s in split]
        return _padded(rows, shape, dtype, fill)

    def counts(family: str) -> jax.Array:
        return jnp.array([len(getattr(s, family)) for s in split], dtype=jnp.int32)

    mask = (n_params,)
    return CompiledScalarGraphs(
        num_graphs=len(split),
        n_params=n_params,
        node_phases=NodePhases(
            phases=padded_field("nodes", 0),
            params=padded_field("nodes", 1, mask),
            counts=counts("nodes"),
        ),
        phase_terms=PhaseTerms(
            coeffs=padded_field("phases", 0), params=padded_field("phases", 1, mask)
        ),
        pi_products=PiProducts(
            psi_params=padded_field("pi_pairs", 0, mask),
            phi_params=padded_field("pi_pairs", 1, mask),
        ),
        phase_pairs=PhasePairs(
            alpha=padded_field("phase_pairs", 0),
            alpha_params=padded_field("phase_pairs", 2, mask),
            beta=padded_field("phase_pairs", 1),
            beta_params=padded_field("phase_pairs", 3, mask),
            counts=counts("phase_pairs"),
        ),
        float_terms=FloatTerms(
            table=padded_field("floats", 0, (4,), np.complex64, fill=1),
            a_params=padded_field("floats", 1, mask),
            b_params=padded_field("floats", 2, mask),
        ),
        prefactor=_scalar_prefactor(split),
    )
