"""Term-family modules for compiled scalar graphs.

Each compiled scalar is the product of four exact term families (``NodePhases``,
``PhaseTerms``, ``PiProducts``, ``PhasePairs``) and a floating-point family
(``FloatTerms``, for angles that are not multiples of π/4), times a per-graph
prefactor (``ScalarPrefactor``). This module defines the families as ``equinox.Module``
records, bundles the shared phase tables, and gives each family an ``evaluate``
method that turns a batch of binary parameter values into an
``ExactScalarArray`` (or, for ``FloatTerms``, a complex array).

Downstream, ``compile.py`` builds instances of these classes from ``zxparam``
scalars (``compile_scalars``), and ``evaluate.py`` orchestrates the products.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from tsim.core.exact_scalar import ExactScalarArray
from tsim.utils.linalg import matmul_gf2

# Powers of ω = e^(iπ/4). UNIT_PHASES[k] is the exact 4-coefficient
# representation of ω^k.
UNIT_PHASES = jnp.array(
    [
        [1, 0, 0, 0],  # omega^0 = 1
        [0, 1, 0, 0],  # omega^1
        [0, 0, 1, 0],  # omega^2 = i
        [0, 0, 0, -1],  # omega^3
        [-1, 0, 0, 0],  # omega^4 = -1
        [0, -1, 0, 0],  # omega^5
        [0, 0, -1, 0],  # omega^6 = -i
        [0, 0, 0, 1],  # omega^7
    ],
    dtype=jnp.int32,
)

# Lookup table for exact scalars (1 + ω^k).
_ONE_PLUS_PHASES = UNIT_PHASES.at[:, 0].add(1)

_IDENTITY = jnp.array([1, 0, 0, 0], dtype=jnp.int32)


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

    def evaluate(self, param_vals: Array) -> ExactScalarArray:
        """Evaluate Π (1 + ω^(4·parity + phase)) per graph, batched over param_vals.

        Args:
            param_vals: Binary parameter values, shape ``(batch, n_params)``.

        Returns:
            ``ExactScalarArray`` of shape ``(batch, num_graphs)``.

        """
        rowsum = matmul_gf2(self.params, param_vals)
        phase_idx = (4 * rowsum + self.phases) % 8

        term_vals = _ONE_PLUS_PHASES[phase_idx]
        mask = jnp.arange(self.phases.shape[1])[None, :] < self.counts[:, None]
        term_vals = jnp.where(mask[..., None], term_vals, _IDENTITY)

        return ExactScalarArray(term_vals).prod(axis=-1)


class PhaseTerms(eqx.Module):
    """Product of ``exp(i·π·α·⊕params)`` terms, each angle ``π·α`` a multiple of π/4.

    Each term is stored as ``k = 4·α mod 8 ∈ 0..7`` and evaluates to
    ``ω^(k·⊕params)`` with ``ω = e^(iπ/4)``. Terms sharing a parameter bitstring
    have been combined to a single stored coefficient (see ``compile_scalars``).

    Padded slots use ``0`` (the additive identity for phase sums), so padded
    entries contribute nothing to the summed exponent.

    Shapes are ``(num_graphs, max_terms)`` except ``params`` which is
    ``(num_graphs, max_terms, n_params)``.
    """

    coeffs: Array  # uint8, values 0-7 in eighth turns (0 = padding)
    params: Array  # uint8, parameter parity bitmasks

    def evaluate(self, param_vals: Array) -> ExactScalarArray:
        """Evaluate ω^(Σ coeffs · parity) per graph, batched over param_vals.

        Args:
            param_vals: Binary parameter values, shape ``(batch, n_params)``.

        Returns:
            ``ExactScalarArray`` of shape ``(batch, num_graphs)``.

        """
        rowsum = matmul_gf2(self.params, param_vals)
        phase_idx = (rowsum * self.coeffs) % 8
        total_phase = jnp.sum(phase_idx, axis=-1) % 8
        return ExactScalarArray(UNIT_PHASES[total_phase])


class PiProducts(eqx.Module):
    """Product of ``(-1)^(ψ · φ)`` terms, with ψ and φ each a parameter parity.

    Each side is encoded as a parameter bitmask (zxparam expands constant bits
    into other terms). Padded slots use ``0`` everywhere; a padded term
    contributes ``(-1)^0 = 1`` to the product.

    Shapes are ``(num_graphs, max_terms, n_params)``.
    """

    psi_params: Array  # uint8, parameter parity bitmask for ψ
    phi_params: Array  # uint8, parameter parity bitmask for φ

    def evaluate(self, param_vals: Array) -> ExactScalarArray:
        """Evaluate Π (-1)^(ψ·φ) per graph as a real ±1 exact scalar.

        Args:
            param_vals: Binary parameter values, shape ``(batch, n_params)``.

        Returns:
            ``ExactScalarArray`` of shape ``(batch, num_graphs)``, with values
            in {+1, -1} represented exactly.

        """
        psi = matmul_gf2(self.psi_params, param_vals)
        phi = matmul_gf2(self.phi_params, param_vals)

        exponent = (psi * phi) % 2
        sum_exponents = jnp.sum(exponent, axis=-1) % 2

        # (1 - 2·bit) ∈ {+1, -1}; promote to the 4-coefficient ExactScalar basis.
        summands_exact = (1 - 2 * sum_exponents)[..., None] * _IDENTITY
        return ExactScalarArray(summands_exact)


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

    def evaluate(self, param_vals: Array) -> ExactScalarArray:
        """Evaluate Π (1 + ω^α + ω^β - ω^(α+β)) per graph, batched.

        Args:
            param_vals: Binary parameter values, shape ``(batch, n_params)``.

        Returns:
            ``ExactScalarArray`` of shape ``(batch, num_graphs)``.

        """
        rowsum_a = matmul_gf2(self.alpha_params, param_vals)
        rowsum_b = matmul_gf2(self.beta_params, param_vals)

        alpha = (self.alpha + rowsum_a * 4) % 8
        beta = (self.beta + rowsum_b * 4) % 8
        gamma = (alpha + beta) % 8

        term_vals = (
            _IDENTITY + UNIT_PHASES[alpha] + UNIT_PHASES[beta] - UNIT_PHASES[gamma]
        )
        mask = jnp.arange(self.alpha.shape[1])[None, :] < self.counts[:, None]
        term_vals = jnp.where(mask[..., None], term_vals, _IDENTITY)

        return ExactScalarArray(term_vals).prod(axis=-1)


class ScalarPrefactor(eqx.Module):
    """Per-graph static scalar prefactor — independent of parameter values.

    Each graph's value is the product of the four exact term families and
    ``FloatTerms`` times the prefactor ``floatfactor · 2^power2`` (which includes
    any global phase that is a multiple of π/4), times the graph's complex entry
    of ``approximate_floatfactors``. That entry collects the parameter-free
    factors whose angles are not multiples of π/4 (see ``_split_scalar`` in
    ``compile.py``): a global phase whose denominator is outside ``{1, 2, 4}``,
    and parameter-free ``NodeTerm``s and ``PhasePair``s with such angles. It is
    ``1`` when a graph has none of them.

    This class is pure data; the final fold-in with the term-family product
    (which branches on ``has_approximate_floatfactors`` and on
    ``CompiledScalarGraphs.has_float_terms``) lives in ``evaluate.py``.
    """

    floatfactor: Array  # int32, shape (num_graphs, 4) — exact dyadic (a,b,c,d)
    power2: Array  # int32, shape (num_graphs,) — exponent on the 2^(·) scaling
    approximate_floatfactors: Array  # complex64, shape (num_graphs,)
    has_approximate_floatfactors: bool = eqx.field(static=True)


class FloatTerms(eqx.Module):
    """Product of terms whose angles are not multiples of π/4, in floating point.

    Each term has up to two parameter parities ``p_a`` and ``p_b`` and a table of
    its four values, indexed by ``p_a + 2·p_b``. Padded slots hold a table of ones
    and empty masks.

    Shapes are ``(num_graphs, max_terms, 4)`` for ``table`` and
    ``(num_graphs, max_terms, n_params)`` for the masks.
    """

    table: Array  # complex64
    a_params: Array  # uint8, parity mask of p_a
    b_params: Array  # uint8, parity mask of p_b

    def evaluate(self, param_vals: Array) -> Array:
        """Evaluate the product per graph, batched over ``param_vals``.

        Args:
            param_vals: Binary parameter values, shape ``(batch, n_params)``.

        Returns:
            Complex array of shape ``(batch, num_graphs)``.

        """
        pa = matmul_gf2(self.a_params, param_vals).astype(jnp.int32)
        pb = matmul_gf2(self.b_params, param_vals).astype(jnp.int32)
        index = pa + 2 * pb
        table = jnp.broadcast_to(self.table[None], (*index.shape, 4))
        values = jnp.take_along_axis(table, index[..., None], axis=-1)[..., 0]
        return jnp.prod(values, axis=-1)
